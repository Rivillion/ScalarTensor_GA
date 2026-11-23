using Random
include(joinpath(@__DIR__, "src", "run_ga.jl"))

# Cluster-oriented run configuration. Edit this file instead of changing the
# defaults in src/run_ga.jl. Set this to false for a fully random initial
# population with no later seed reinjection.
const ENABLE_PHYSICAL_SEEDING = true

function configured_seed_settings()
    ENABLE_PHYSICAL_SEEDING || return SeedSettings()

    return SeedSettings(
        guesses=[
            #GuessSpec(
            #    label="LCDM",
            #    F="1.0",
            #    U="0.0",
            #),
            GuessSpec(
                label="linear coupling and quadratic potential",
                F="1.01 + 0.1 * Phi",
                U="0.1 * Phi * Phi",
            ),
            GuessSpec(
                label="Dimitrios and Achilleas",
                F="1.0 + 0.1 * Phi * Phi * exp(-0.1 * Phi)",
                U="0.1 * exp(-0.1 * Phi * Phi)",
            ),
        ],
        initial_fraction=0.05,
        neighbor_mutations_min=1,
        neighbor_mutations_max=3,
        mutation_retry_limit=20,
        require_function_sanity=true,
        reject_duplicate_guesses=true,
        reinject_after=50,
        reinject_interval=25,
        reinjection_fraction=0.02,
    )
end

function validate_probability(name::AbstractString, value::Float64)
    isfinite(value) && 0.0 <= value <= 1.0 || throw(ArgumentError(
        "$name must be a finite number between 0 and 1; got $value"))
    return value
end

function configured_run_config(;
    seed::UInt64=rand(RandomDevice(), UInt64),
    crossover_rate::Float64=0.75,
    mutation_rate::Float64=0.50,
    output_root::String=get(ENV, "GA_OUTPUT_ROOT", joinpath(@__DIR__, "runs")),
    penalty_mode::Symbol=Symbol(get(ENV, "GA_PENALTY_MODE", "functional_only")),
)
    validate_probability("crossover rate", crossover_rate)
    validate_probability("mutation rate", mutation_rate)

    return RunConfig(
        ga=GASettings(
            Npops=1000,
            Ngens=2000,
            tour_size=4,
            selection_rate=0.50,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_size=15,
            likelihoods=[
                :PantheonPlus_SH0ES_muG,
                :DESI_BAO,
                :Planck_Zhai,
                :Growth_RSD,
                :PPN_gamma_Cassini,
                :Gdot_LLR,
                :BBN_G,
            ],
        ),
        seeding=configured_seed_settings(),
        penalty=PenaltySettings(
            mode=penalty_mode,
            fisher_pruning=false,
        ),
        physics=default_physics_settings(),
        rng=RNGConfig(
            seed=seed,
        ),
        storage=StorageConfig(
            root=output_root,
            checkpoint=CheckpointConfig(
                interval_generations=25,
                retain=2,
                on_error=:warn,
            ),
            top_k=nothing,
            save_final_population=false,
        ),
    )
end

function configured_run(; kwargs...)
    config = configured_run_config(; kwargs...)
    print_run_configuration(config)
    return run_ga(config)
end

# --auto: one output folder per run (the SLURM script gives every array task its own). The task starts, resumes or
# skips by itself, so submitting the same job again continues whatever is unfinished.
const COMPLETED_MARKER = "COMPLETED"   # written in the output folder once its run has finished every generation
const ABANDONED_DIRECTORY = "abandoned"
const EXIT_NOT_FINISHED = 75           # exit code of a run that stopped early and can be resumed

"""
    run_directories(root) -> Vector{String}

The run folders directly inside the output folder `root`: its subfolders that hold a `config.toml`.

Generated with Claude Fable 5.2.
"""
function run_directories(root::AbstractString)
    isdir(root) || return String[]
    return [path for path in readdir(root; join=true)
            if isdir(path) && isfile(joinpath(path, "config.toml"))]
end

"""
    run_status(run_dir) -> String

The `status` of a run from its `metadata.toml`: "completed", "interrupted" or "failed" once it was finalized, and
"running" before that (also for a run that was killed).

Generated with Claude Fable 5.2.
"""
function run_status(run_dir::AbstractString)
    try
        return String(get(TOML.parsefile(joinpath(run_dir, "metadata.toml")), "status", "running"))
    catch
        return "running"
    end
end

"""
    newest_valid_checkpoint(run_dir) -> Union{Nothing,NamedTuple}

The checkpoint of `run_dir` with the highest generation that passes `validate_checkpoint` (checksum and contents), as
`(path, generation, run_dir)`, or `nothing` if there is none. A damaged or half-written checkpoint is reported and
skipped, so the previous one is used.

Generated with Claude Fable 5.2.
"""
function newest_valid_checkpoint(run_dir::AbstractString)
    directory = joinpath(run_dir, "checkpoints")
    isdir(directory) || return nothing
    found = Tuple{Int,Bool,String}[]
    for name in readdir(directory)
        m = match(r"^checkpoint-g(\d+)(-final)?\.h5$", name)
        m === nothing || push!(found, (parse(Int, m[1]), m[2] !== nothing, joinpath(directory, name)))
    end
    for (generation, _, path) in sort(found; rev=true)
        try
            validate_checkpoint(path)
            return (; path, generation, run_dir=String(run_dir))
        catch error
            println("Skipping unusable checkpoint $path: $(sprint(showerror, error))")
        end
    end
    return nothing
end

"""
    settings_differences(a::RunConfig, b::RunConfig) -> Vector{String}

The `config.toml` entries (as "section.key") in which two configurations differ, leaving out the output folder.

Generated with Claude Fable 5.2.
"""
function settings_differences(a::RunConfig, b::RunConfig)
    da, db = config_dict(a), config_dict(b)
    differences = String[]
    for section in sort!(collect(union(keys(da), keys(db))))
        sa, sb = get(da, section, Dict()), get(db, section, Dict())
        for key in sort!(collect(union(keys(sa), keys(sb))))
            section == "storage" && key == "root" && continue
            isequal(get(sa, key, nothing), get(sb, key, nothing)) || push!(differences, "$section.$key")
        end
    end
    return differences
end

"""
    auto_run(config::RunConfig) -> Symbol

Start, resume or skip the run of the output folder `config.storage.root`, whichever applies:

- a run there is already completed: write the `COMPLETED` marker and do nothing else;
- a run there has a valid checkpoint: resume it from the newest one (it must have been made with the settings of
  `config`, otherwise this throws);
- otherwise: start a new run.

Run folders that cannot be resumed (killed before their first checkpoint) are moved to `abandoned/`. Returns
`:completed`, after writing the marker, or `:stopped` if the run ended early (stop file) and can be resumed.

Generated with Claude Fable 5.2.
"""
function auto_run(config::RunConfig)
    root = abspath(config.storage.root)
    marker = joinpath(root, COMPLETED_MARKER)
    runs = run_directories(root)
    finished = findfirst(run -> run_status(run) == "completed", runs)
    if finished !== nothing
        touch(marker)
        println("Already completed, nothing to do: $(runs[finished])")
        return :completed
    end

    best = nothing
    for run in runs
        found = newest_valid_checkpoint(run)
        found === nothing && continue
        (best === nothing || found.generation > best.generation) && (best = found)
    end
    for run in runs
        best !== nothing && run == best.run_dir && continue
        target = joinpath(root, ABANDONED_DIRECTORY, basename(run))
        mkpath(dirname(target))
        mv(run, target)
        println("No usable checkpoint in $run: moved to $target")
    end

    if best === nothing
        println("Starting a new run in $root")
        print_run_configuration(config)
        run_ga(config)
    else
        stored, _ = _read_config_for_checkpoint(best.path)
        differences = settings_differences(stored, config)
        isempty(differences) || throw(ArgumentError(
            "the run in $(best.run_dir) was made with other settings ($(join(differences, ", "))). " *
            "Restore them to resume it, or use a new output folder (a new CAMPAIGN name in the SLURM file)."))
        println("Resuming from generation $(best.generation) of $(stored.ga.Ngens): $(best.path)")
        print_run_configuration(stored)
        resume_run(best.path)
    end

    if any(run -> run_status(run) == "completed", run_directories(root))
        touch(marker)
        return :completed
    end
    return :stopped
end

function print_run_configuration(config::RunConfig)
    println("GA seed: $(config.rng.seed)")
    println("Crossover rate: $(config.ga.crossover_rate)")
    println("Mutation rate: $(config.ga.mutation_rate)")
    println("Output root: $(abspath(config.storage.root))")
    println("Penalty mode: $(config.penalty.mode)")
    return nothing
end

const CONFIGURED_USAGE = """
usage:
  julia --project=. ga_run.jl [OPTIONS]
  julia --project=. ga_run.jl --auto --output-root PATH [OPTIONS]
  julia --project=. ga_run.jl --resume CHECKPOINT

options:
  --seed UINT64              RNG seed (default: draw from RandomDevice)
  --crossover-rate FLOAT     crossover probability in [0, 1] (default: 0.75)
  --mutation-rate FLOAT      per-child mutation probability in [0, 1] (default: 0.50)
  --output-root PATH         output directory (default: GA_OUTPUT_ROOT or ./runs)
  --penalty-mode MODE        penalty mode (default: GA_PENALTY_MODE or functional_only)
  --auto                     start, resume or skip the run of --output-root, whichever applies: resume from its
                             newest valid checkpoint, start a new run if there is none, do nothing if it is
                             completed. Exit code 75 means the run stopped early; run the same command again.
  --dry-run                  validate and print the configuration without running the GA
  -h, --help                 show this help
"""

function option_value(args, index::Int, option::String)
    index < length(args) || throw(ArgumentError("$option requires a value\n$CONFIGURED_USAGE"))
    return args[index + 1]
end

function parse_configured_args(args)
    if length(args) == 2 && args[1] == "--resume"
        return (resume_path=args[2], help=false, dry_run=false, auto=false, kwargs=nothing)
    elseif "--resume" in args
        throw(ArgumentError("--resume cannot be combined with other options\n$CONFIGURED_USAGE"))
    end

    seed = rand(RandomDevice(), UInt64)
    crossover_rate = 0.75
    mutation_rate = 0.50
    output_root = get(ENV, "GA_OUTPUT_ROOT", joinpath(@__DIR__, "runs"))
    penalty_mode = Symbol(get(ENV, "GA_PENALTY_MODE", "functional_only"))
    dry_run = false
    help = false
    auto = false

    index = 1
    while index <= length(args)
        argument = args[index]
        if argument == "--seed"
            seed = parse(UInt64, option_value(args, index, argument))
            index += 2
        elseif argument == "--crossover-rate"
            crossover_rate = parse(Float64, option_value(args, index, argument))
            index += 2
        elseif argument == "--mutation-rate"
            mutation_rate = parse(Float64, option_value(args, index, argument))
            index += 2
        elseif argument == "--output-root"
            output_root = option_value(args, index, argument)
            index += 2
        elseif argument == "--penalty-mode"
            penalty_mode = Symbol(option_value(args, index, argument))
            index += 2
        elseif argument == "--auto"
            auto = true
            index += 1
        elseif argument == "--dry-run"
            dry_run = true
            index += 1
        elseif argument in ("-h", "--help")
            help = true
            index += 1
        else
            throw(ArgumentError("unknown argument: $argument\n$CONFIGURED_USAGE"))
        end
    end

    kwargs = (; seed, crossover_rate, mutation_rate, output_root, penalty_mode)
    return (; resume_path=nothing, help, dry_run, auto, kwargs)
end

function main_configured(args=ARGS)
    options = parse_configured_args(args)
    if options.help
        print(CONFIGURED_USAGE)
        return nothing
    elseif options.resume_path !== nothing
        return resume_run(options.resume_path)
    end

    config = configured_run_config(; options.kwargs...)
    if options.dry_run
        validate_minimize_settings(config.minimizer)
        validate_run_config(config)
    end
    if options.auto && !options.dry_run
        # Exit code 75 tells the SLURM script that the run stopped early and can be resumed.
        auto_run(config) == :completed || exit(EXIT_NOT_FINISHED)
        return nothing
    end
    print_run_configuration(config)
    options.dry_run && return config
    return run_ga(config)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main_configured()
