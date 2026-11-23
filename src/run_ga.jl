using Base.Threads
using Symbolics
using Logging
using Statistics

disable_logging(Logging.Info)

include("GA.jl")
include("minimize.jl")

Base.@kwdef struct GASettings
    Npops::Int = 2500
    Ngens::Int = 2500
    tour_size::Int = 4
    selection_rate::Float64 = 0.50
    crossover_rate::Float64 = 0.75
    mutation_rate::Float64 = 0.50
    max_size::Int = 15
    likelihoods::Vector{Symbol} = [
        :PantheonPlus_SH0ES,
        :DESI_BAO,
        :Planck_Zhai,
        :Growth_RSD,
    ]
end

function replacement_count(Npops::Int, selection_rate::Float64)::Int
    Npops >= 3 || throw(ArgumentError("Npops must be at least 3; got $Npops."))
    0.0 < selection_rate < 1.0 || throw(ArgumentError(
        "selection_rate must be strictly between 0 and 1; got $selection_rate."))
    count = 2 * round(Int, selection_rate * Npops / 2)
    2 <= count <= Npops - 1 || throw(ArgumentError(
        "selection_rate=$selection_rate with Npops=$Npops produces " *
        "replacement_count=$count; choose an even value from 2 to $(Npops - 1)."))
    return count
end

function validate_selection_settings(settings::GASettings)
    replacement_count(settings.Npops, settings.selection_rate)
    1 <= settings.tour_size <= settings.Npops || throw(ArgumentError(
        "tour_size must be between 1 and Npops=$(settings.Npops)."))
    return nothing
end

function validate_seed_settings(seed_settings::SeedSettings, ga_settings::GASettings)
    0.0 <= seed_settings.initial_fraction <= 1.0 || throw(ArgumentError(
        "initial seed fraction must be between zero and one"))
    0.0 <= seed_settings.reinjection_fraction <= 1.0 || throw(ArgumentError(
        "seed reinjection fraction must be between zero and one"))
    seed_settings.neighbor_mutations_min >= 1 || throw(ArgumentError(
        "neighbor_mutations_min must be at least one"))
    seed_settings.neighbor_mutations_max >= seed_settings.neighbor_mutations_min ||
        throw(ArgumentError("neighbor_mutations_max is smaller than its minimum"))
    seed_settings.mutation_retry_limit >= 1 || throw(ArgumentError(
        "mutation_retry_limit must be at least one"))
    seed_settings.reinject_after === nothing || seed_settings.reinject_after >= 1 ||
        throw(ArgumentError("reinject_after must be nothing or positive"))
    seed_settings.reinject_interval >= 1 || throw(ArgumentError(
        "reinject_interval must be at least one"))
    length(seed_settings.guesses) <= ga_settings.Npops || throw(ArgumentError(
        "the number of physical guesses exceeds Npops"))
    isempty(seed_settings.guesses) && seed_settings.reinject_after !== nothing &&
        throw(ArgumentError("seed reinjection requires physical guesses"))
    return nothing
end

const DATA_PATHS = Dict{Symbol,Tuple{Vararg{String}}}(
    :PantheonPlus_SH0ES => (
        joinpath(_PROJECT_ROOT, "data", "Pantheon+SH0ES.dat"),
        joinpath(_PROJECT_ROOT, "data", "Pantheon+SH0ES_STAT+SYS.cov")),
    :DESI_BAO => (
        joinpath(_PROJECT_ROOT, "data", "desi_gaussian_bao_ALL_GCcomb_mean.txt"),
        joinpath(_PROJECT_ROOT, "data", "desi_gaussian_bao_ALL_GCcomb_cov.txt")),
    :Growth_RSD => (
        joinpath(_PROJECT_ROOT, "data", "growth_data.txt"),
        joinpath(_PROJECT_ROOT, "data", "wigglez_cov.txt"),
        joinpath(_PROJECT_ROOT, "data", "sdss_DR12_LRG_FSBAO_DMDHfs8_covtot.txt")),
)

function load_likelihoods(names::Vector{Symbol})
    likelihoods = AbstractLikelihood[]
    for name in names
        likelihood = if name == :Planck_Zhai
            PlanckCompressed_Zhai2018()
        elseif name == :GdotLikelihood
            GdotLikelihood()
        elseif name == :MuGLikelihood
            MuGLikelihood()
        elseif name == :PPN_gamma_Cassini
            PPNGammaLikelihood()
        elseif name == :Gdot_LLR
            LLRGdotLikelihood()
        elseif name == :BBN_G
            BBNGLikelihood()
        elseif name == :PantheonPlus_SH0ES
            PantheonPlusSH0ES(DATA_PATHS[:PantheonPlus_SH0ES]...)
        elseif name == :PantheonPlus_SH0ES_muG
            PantheonPlusSH0ES_muG(PantheonPlusSH0ES(DATA_PATHS[:PantheonPlus_SH0ES]...))
        elseif name == :DESI_BAO
            DESI_BAO(DATA_PATHS[:DESI_BAO]...)
        elseif name == :Growth_RSD
            GrowthRateLikelihood(DATA_PATHS[:Growth_RSD]...)
        else
            throw(ArgumentError("unknown likelihood name: $name"))
        end
        push!(likelihoods, likelihood)
    end
    return likelihoods
end

include("storage.jl")

const GA_MODEL_ID = "nonminimal-scalar-tensor-recovery-v1"
const GA_SCHEMA_ID = "optimizer-recovery-v1"

default_physics_settings() = PhysicsSettings(model_id=GA_MODEL_ID)
default_run_config() = RunConfig(physics=default_physics_settings())

function initialize_run_storage_with_schema(config::RunConfig)
    metadata = collect_run_metadata(config)
    metadata["schema_id"] = GA_SCHEMA_ID
    metadata["recovery_diagnostics"] = "events.jsonl:generation_diagnostics"
    return initialize_run_storage(config, metadata)
end

struct EvaluationBatchDiagnostics
    initial_failures::Int
    recovery_attempts::Int
    recovery_successes::Int
    accepted::Int
    feasibility_evals::Int
    objective_evals::Int
    terminal_failures::Dict{Symbol,Int}
end

function EvaluationBatchDiagnostics(results)
    failures = Dict{Symbol,Int}()
    for result in results
        result.accepted && continue
        failures[result.final_status] = get(failures, result.final_status, 0) + 1
    end
    return EvaluationBatchDiagnostics(
        count(result -> result.initial_status != :valid, results),
        count(result -> result.recovery_probes > 0, results),
        count(result -> result.recovery_succeeded, results),
        count(result -> result.accepted, results),
        sum(result.feasibility_evals for result in results),
        sum(result.objective_evals for result in results),
        failures,
    )
end

function diagnostics_string(diagnostics::EvaluationBatchDiagnostics)
    ordered = sort!(collect(diagnostics.terminal_failures); by=first)
    failures = join(("$(key)=$(value)" for (key, value) in ordered), ",")
    return "initial_failures=$(diagnostics.initial_failures)," *
        "recovery_attempts=$(diagnostics.recovery_attempts)," *
        "recovery_successes=$(diagnostics.recovery_successes)," *
        "accepted=$(diagnostics.accepted)," *
        "feasibility_evals=$(diagnostics.feasibility_evals)," *
        "objective_evals=$(diagnostics.objective_evals),terminal_failures={$failures}"
end

function build_thread_resources(likelihoods)
    dummy_F, dummy_U = compile_tree_lcdm()
    dummy_p = CosmoParams(0.295, 0.0, dummy_F, dummy_U)
    caches = [setup_cosmo_cache(dummy_p, 0.685, 0.0) for _ in 1:Threads.maxthreadid()]
    # Per-thread scratch buffers; the read-only data and covariance factors are shared (see `thread_copy`).
    likes = [AbstractLikelihood[thread_copy(like) for like in likelihoods]
             for _ in 1:Threads.maxthreadid()]
    return likes, caches
end

function evaluate_individuals!(individuals::Vector{Individual}, thread_likes,
    thread_caches, config::RunConfig)
    fitted = Vector{FittedCandidate}(undef, length(individuals))
    results = Vector{Any}(undef, length(individuals))
    with_logger(SimpleLogger(stderr, Logging.Error)) do
        for i in eachindex(individuals)
            individuals[i] = simplify_individual(individuals[i])
        end
        Threads.@threads :static for i in eachindex(individuals)
            thread = Threads.threadid()
            result = optimize_individual!(individuals[i], thread_likes[thread],
                thread_caches[thread]; settings=config.minimizer,
                penalty_settings=config.penalty)
            results[i] = result
            fitted[i] = fitted_candidate(result, config.penalty)
        end
    end
    # Once per batch (one generation), after all threads are done with Symbolics.
    compact_symbolic_caches!()
    return (candidates=fitted, diagnostics=EvaluationBatchDiagnostics(results))
end

function feasibility_rank(termination::Symbol)
    text = String(termination)
    occursin("likelihood_invalid", text) && return 1
    occursin("full_background_failed", text) && return 2
    occursin("shooting_refinement_failed", text) && return 3
    occursin("shooting_no_bracket", text) && return 4
    occursin("shooting_no_physical_trial", text) && return 5
    occursin("function_invalid", text) && return 6
    return 7
end

function candidate_selection_key(candidate::FittedCandidate)
    digest = candidate_digest(candidate)
    if candidate.accepted
        return (0, candidate.description_length, Float64(candidate.individual.complexity), digest)
    end
    return (feasibility_rank(candidate.termination),
        Float64(candidate.individual.complexity), candidate.description_length, digest)
end

function _tournament_index(rng::AbstractRNG, keys, eligible::Vector{Int},
    tour_size::Int; select_best::Bool)
    isempty(eligible) && throw(ArgumentError("tournament has no eligible candidates"))
    sample = eligible[randperm(rng, length(eligible))[1:tour_size]]
    position = select_best ? argmin(@view keys[sample]) : argmax(@view keys[sample])
    return sample[position]
end

function _distinct_tournaments(rng::AbstractRNG, keys, eligible::Vector{Int}, count::Int,
    tour_size::Int; select_best::Bool)
    count <= length(eligible) || throw(ArgumentError("not enough eligible candidates"))
    available = copy(eligible)
    selected = Int[]
    sizehint!(selected, count)
    for _ in 1:count
        size = Base.min(tour_size, length(available))
        index = _tournament_index(rng, keys, available, size; select_best)
        push!(selected, index)
        deleteat!(available, something(findfirst(==(index), available)))
    end
    return selected
end

"""Hybrid selection: invalid lineages only fill a shortage of accepted parents."""
function select_replacement_indices_by_key(rng::AbstractRNG,
    population::Vector{FittedCandidate}, count::Int, tour_size::Int)
    population_size = length(population)
    2 <= count <= population_size - 1 || throw(ArgumentError(
        "replacement count must be between 2 and Npops - 1"))
    iseven(count) || throw(ArgumentError("replacement count must be even"))
    keys = candidate_selection_key.(population)
    accepted = findall(candidate -> candidate.accepted, population)
    invalid = findall(candidate -> !candidate.accepted, population)
    parent_pool = copy(accepted)
    if length(parent_pool) < count
        ordered_invalid = sort(invalid; by=index -> keys[index])
        append!(parent_pool, @view ordered_invalid[1:(count - length(parent_pool))])
    end
    parents = _distinct_tournaments(rng, keys, parent_pool, count, tour_size;
        select_best=true)
    shuffle!(rng, parents)

    champion = isempty(accepted) ? nothing : accepted[argmin(@view keys[accepted])]
    loser_pool = champion === nothing ? collect(1:population_size) :
        [index for index in 1:population_size if index != champion]
    losers = _distinct_tournaments(rng, keys, loser_pool, count, tour_size;
        select_best=false)
    return (parents=parents, losers=losers, champion=champion)
end

function _accepted_or_fallback(population::Vector{FittedCandidate}; criterion=:score)
    accepted = filter(candidate -> candidate.accepted, population)
    if !isempty(accepted)
        key = criterion == :chi2 ? (candidate -> candidate.chi2) :
            (candidate -> candidate.description_length)
        return accepted[argmin(key(candidate) for candidate in accepted)]
    end
    return population[argmin(candidate_selection_key(candidate) for candidate in population)]
end

function update_valid_archive!(archive::Vector{FittedCandidate}, candidates, k::Int)
    accepted = filter(candidate -> candidate.accepted, candidates)
    isempty(accepted) || update_archive!(archive, accepted, k)
    return archive
end

function final_archive(state::GARunState, config::RunConfig)
    accepted = filter(candidate -> candidate.accepted, state.population)
    order = sortperm(accepted; by=candidate -> candidate.description_length)
    count = Base.min(resolved_top_k(config), length(order))
    return copy_fitted.(@view accepted[order[1:count]])
end

function generation_metrics(state::GARunState, generation_best::FittedCandidate,
    previous_best::Float64)
    scores = [candidate.description_length for candidate in state.population]
    best = state.global_best
    improvement = best.accepted && isfinite(previous_best) ?
        Base.max(0.0, previous_best - best.description_length) : 0.0
    previous_ga = isempty(state.history) ? 0 : state.history[end].ga_evaluations
    previous_objective = isempty(state.history) ? 0 : state.history[end].objective_evaluations
    return GenerationMetrics(
        state.completed_generation, state.ga_evaluations, state.ga_evaluations - previous_ga,
        state.objective_evaluations, state.objective_evaluations - previous_objective,
        state.elapsed_seconds, generation_best.chi2, generation_best.description_length,
        generation_best.individual.complexity, best.chi2, best.description_length,
        best.individual.complexity, best.penalty.functional, best.penalty.structural,
        best.penalty.sign, best.penalty.constants, best.penalty.fisher,
        count(candidate -> candidate.accepted, state.population),
        count(candidate -> !candidate.accepted, state.population),
        count(candidate -> !candidate.converged, state.population),
        length(unique(candidate_digest(candidate) for candidate in state.population)),
        median(scores), quantile(scores, 0.10), quantile(scores, 0.90), improvement,
        state.generations_since_improvement, copy(best.param_names), copy(best.best_params))
end

function report_generation(state::GARunState, settings::GASettings)
    best = state.global_best
    label = best.accepted ? "physical champion" : "diagnostic fallback (no accepted model)"
    println("Generation $(state.completed_generation) / $(settings.Ngens) | $label | " *
        "L(D)=$(round(best.description_length, digits=2)); chi2=$(round(best.chi2, digits=2))")
    println("   F(Phi) = $(best.individual.F) and U(Phi) = $(best.individual.U)\n")
end

function _build_initial_individuals(rng::AbstractRNG, ga::GASettings, seeding::SeedSettings)
    exact = build_exact_seeds(seeding.guesses, 1, ga.max_size, seeding)
    requested = round(Int, seeding.initial_fraction * ga.Npops)
    seed_slots = isempty(exact) ? 0 : clamp(requested, length(exact), ga.Npops)
    individuals = Individual[_copy_individual(seed, 1) for seed in exact]
    forbidden = Set(individual_signature(seed) for seed in exact)
    neighbors = 0
    failed_labels = String[]
    for slot in (length(exact) + 1):seed_slots
        index = mod1(slot - length(exact), length(exact))
        variant = mutated_seed_variant(rng, exact[index], 1, ga.max_size, seeding, forbidden)
        if variant === nothing
            push!(failed_labels, seeding.guesses[index].label)
            continue
        end
        push!(individuals, _copy_individual(variant, 1))
        push!(forbidden, individual_signature(variant))
        neighbors += 1
    end
    append!(individuals, (generate_random_individual(rng, ga.max_size, 1)
        for _ in 1:(ga.Npops - length(individuals))))
    !isempty(exact) && shuffle!(rng, individuals)
    return (individuals, exact_count=length(exact), requested_seed_slots=seed_slots,
        neighbors_generated=neighbors, random_count=ga.Npops - seed_slots,
        failed_labels)
end

build_initial_individuals(rng::AbstractRNG, ga::GASettings, seeding::SeedSettings) =
    _build_initial_individuals(rng, ga, seeding).individuals

function _log_batch(storage, generation::Int, diagnostics::EvaluationBatchDiagnostics)
    storage === nothing && return
    log_storage_event(storage, "generation_diagnostics";
        details="generation=$generation," * diagnostics_string(diagnostics))
end

function initialize_run_state(config::RunConfig, thread_likes, thread_caches, started_at;
    storage=nothing)
    rng = Random.Xoshiro(config.rng.seed)
    build = _build_initial_individuals(rng, config.ga, config.seeding)
    batch = evaluate_individuals!(build.individuals, thread_likes, thread_caches, config)
    population = batch.candidates
    generation_best = _accepted_or_fallback(population)
    best_chi2 = _accepted_or_fallback(population; criterion=:chi2)
    archive = FittedCandidate[]
    update_valid_archive!(archive, population, resolved_top_k(config))
    state = GARunState(1, population, copy_fitted(generation_best), copy_fitted(best_chi2),
        archive, GenerationMetrics[], length(population), batch.diagnostics.objective_evals,
        0, time() - started_at, rng)
    _log_batch(storage, 1, batch.diagnostics)
    return state, generation_best
end

function evolve_generation!(state::GARunState, config::RunConfig, thread_likes,
    thread_caches, generation::Int, started_at::Float64, prior_elapsed::Float64;
    storage=nothing)
    previous_best = state.global_best.accepted ? state.global_best.description_length : Inf
    count = replacement_count(config.ga.Npops, config.ga.selection_rate)
    selected = select_replacement_indices_by_key(
        state.rng, state.population, count, config.ga.tour_size)
    individuals = [candidate.individual for candidate in state.population]
    children = generate_children(state.rng, individuals, selected.parents, config.ga, generation)
    if seed_reinjection_due(state.generations_since_improvement, config.seeding)
        exact = build_exact_seeds(config.seeding.guesses, generation,
            config.ga.max_size, config.seeding)
        inject_seed_variants!(state.rng, children, exact, generation,
            config.ga.max_size, config.seeding)
    end
    batch = evaluate_individuals!(children, thread_likes, thread_caches, config)
    next_population = copy(state.population)
    for i in eachindex(selected.losers)
        next_population[selected.losers[i]] = batch.candidates[i]
    end

    generation_best = _accepted_or_fallback(next_population)
    generation_chi2 = _accepted_or_fallback(next_population; criterion=:chi2)
    next_best = state.global_best
    improved = if generation_best.accepted
        !state.global_best.accepted ||
            generation_best.description_length < state.global_best.description_length
    elseif !state.global_best.accepted
        candidate_selection_key(generation_best) < candidate_selection_key(state.global_best)
    else
        false
    end
    improved && (next_best = copy_fitted(generation_best))
    next_chi2 = state.best_chi2
    if generation_chi2.accepted &&
       (!state.best_chi2.accepted || generation_chi2.chi2 < state.best_chi2.chi2)
        next_chi2 = copy_fitted(generation_chi2)
    elseif !state.best_chi2.accepted && !generation_chi2.accepted &&
           candidate_selection_key(generation_chi2) < candidate_selection_key(state.best_chi2)
        next_chi2 = copy_fitted(generation_chi2)
    end

    archive = copy(state.historical_archive)
    update_valid_archive!(archive, batch.candidates, resolved_top_k(config))
    state.population = next_population
    state.global_best = next_best
    state.best_chi2 = next_chi2
    state.historical_archive = archive
    state.generations_since_improvement = improved ? 0 : state.generations_since_improvement + 1
    state.ga_evaluations += length(batch.candidates)
    state.objective_evaluations += batch.diagnostics.objective_evals
    state.completed_generation = generation
    state.elapsed_seconds = prior_elapsed + (time() - started_at)
    _log_batch(storage, generation, batch.diagnostics)
    return generation_best, previous_best
end

function _save_periodic!(storage, state)
    try
        save_snapshot!(storage, state)
        interval = storage.config.storage.checkpoint.interval_generations
        interval !== nothing && state.completed_generation % interval == 0 &&
            save_checkpoint!(storage, state)
    catch error
        storage.degraded = true
        log_storage_event(storage, "periodic_save_failed"; details=sprint(showerror, error))
        storage.config.storage.checkpoint.on_error == :abort && rethrow()
        @warn "Periodic storage failed; retaining the previous checkpoint" exception=(error, catch_backtrace())
    end
end

"""
    stop_requested() -> Bool

Whether the file named by the environment variable `GA_STOP_FILE` exists. `run_ga` checks it after every generation and,
if it does, saves the run as `:interrupted` (final checkpoint and `result.h5`) and returns. The SLURM script creates
the file shortly before the wall-time limit, so no signal has to reach Julia. Without the variable, nothing is checked.

Generated with Claude Fable 5.2.
"""
function stop_requested()
    path = get(ENV, "GA_STOP_FILE", "")
    return !isempty(path) && isfile(path)
end

function run_ga(config::RunConfig=default_run_config(); resume_from=nothing)
    config.physics.model_id == GA_MODEL_ID || throw(ArgumentError(
        "Runs require physics.model_id=$GA_MODEL_ID"))
    validate_minimize_settings(config.minimizer)
    validate_run_config(config)
    println("Loading likelihoods...")
    likelihoods = load_likelihoods(config.ga.likelihoods)
    thread_likes, thread_caches = build_thread_resources(likelihoods)
    started_at = time()
    if resume_from === nothing
        storage = initialize_run_storage_with_schema(config)
        state, generation_best = initialize_run_state(config, thread_likes, thread_caches,
            started_at; storage)
        record_generation!(storage, state, generation_metrics(state, generation_best, Inf))
        _save_periodic!(storage, state)
    else
        loaded = load_checkpoint(resume_from)
        config_fingerprint(config) == config_fingerprint(loaded.config) ||
            throw(ArgumentError("resume configuration differs from checkpoint"))
        storage, state = loaded.storage, loaded.state
        log_storage_event(storage, "run_resumed"; details=resume_from)
    end
    report_generation(state, config.ga)
    prior_elapsed = state.elapsed_seconds
    started_at = time()
    stopped = false
    try
        for generation in (state.completed_generation + 1):config.ga.Ngens
            boundary_rng = copy(state.rng)
            local generation_best, previous_best
            try
                generation_best, previous_best = evolve_generation!(state, config,
                    thread_likes, thread_caches, generation, started_at, prior_elapsed; storage)
            catch
                state.rng = boundary_rng
                rethrow()
            end
            record_generation!(storage, state,
                generation_metrics(state, generation_best, previous_best))
            _save_periodic!(storage, state)
            report_generation(state, config.ga)
            if generation < config.ga.Ngens && stop_requested()
                stopped = true
                break
            end
        end
        finalize_run!(storage, state; status=stopped ? :interrupted : :completed)
    catch error
        status = error isa InterruptException ? :interrupted : :failed
        state.elapsed_seconds = prior_elapsed + (time() - started_at)
        try
            finalize_run!(storage, state; status)
        catch save_error
            @error "Unable to save final state" exception=(save_error, catch_backtrace())
        end
        rethrow()
    end
    if stopped
        println("Stop requested ($(ENV["GA_STOP_FILE"])): run saved as interrupted at generation " *
            "$(state.completed_generation) of $(config.ga.Ngens). Resume it from its final checkpoint.")
    end
    if state.global_best.accepted
        println("Final physical champion: L(D)=$(state.global_best.description_length), " *
            "chi2=$(state.global_best.chi2)")
    else
        println("No accepted physical candidate was found; saved diagnostic fallback only.")
    end
    println("Results: $(joinpath(storage.run_dir, "result.h5"))")
    return state.global_best
end

function validate_minimize_settings(settings::MinimizeSettings)
    settings.time_limit > 0.0 || throw(ArgumentError("time_limit must be positive"))
    settings.g_tol > 0.0 || throw(ArgumentError("g_tol must be positive"))
    settings.invalid_objective > settings.rejection_objective > 0.0 ||
        throw(ArgumentError("require invalid_objective > rejection_objective > 0"))
    0.0 <= settings.recovery_time_fraction <= 1.0 || throw(ArgumentError(
        "recovery_time_fraction must be between zero and one"))
    0 <= settings.recovery_max_probes <= 8 || throw(ArgumentError(
        "recovery_max_probes must be between zero and eight"))
    settings.recovery_enabled && settings.recovery_max_probes == 0 && throw(ArgumentError(
        "recovery_max_probes must be positive when recovery is enabled"))
    return nothing
end

function main(args=ARGS)
    resume_path = nothing
    interval = nothing
    seed = RNGConfig().seed
    top_k = nothing
    root = "runs"
    penalty_mode = :functional_only
    i = 1
    while i <= length(args)
        argument = args[i]
        if argument == "--resume"
            i += 1; resume_path = args[i]
        elseif argument == "--checkpoint-interval"
            i += 1
            interval = lowercase(args[i]) in ("off", "none", "disabled") ?
                nothing : parse(Int, args[i])
        elseif argument == "--seed"
            i += 1; seed = parse(UInt64, args[i])
        elseif argument == "--top-k"
            i += 1; top_k = parse(Int, args[i])
        elseif argument == "--output-root"
            i += 1; root = args[i]
        elseif argument == "--penalty-mode"
            i += 1; penalty_mode = Symbol(args[i])
        else
            throw(ArgumentError("unknown argument: $argument"))
        end
        i += 1
    end
    resume_path !== nothing && return resume_run(resume_path)
    storage = StorageConfig(root=root,
        checkpoint=CheckpointConfig(interval_generations=interval), top_k=top_k)
    config = RunConfig(penalty=PenaltySettings(mode=penalty_mode),
        physics=default_physics_settings(), rng=RNGConfig(seed=seed), storage=storage)
    return run_ga(config)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
