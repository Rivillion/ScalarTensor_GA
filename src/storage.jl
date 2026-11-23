# All persistence/checkpoint functions in this file were generated with GPT 5.6.
# See docs/AI_GENERATED_FUNCTIONS.md for the function-by-function inventory.
using Dates
using HDF5
using Random
using SHA
using Statistics
using TOML
using UUIDs

const RUN_FORMAT_ID = "julia-ga-st"
const RUN_SCHEMA_MAJOR = 1
const RUN_SCHEMA_MINOR = 0

Base.@kwdef struct CheckpointConfig
    interval_generations::Union{Nothing,Int} = nothing
    retain::Int = 2
    on_error::Symbol = :warn
end

Base.@kwdef struct StorageConfig
    root::String = "runs"
    checkpoint::CheckpointConfig = CheckpointConfig()
    top_k::Union{Nothing,Int} = nothing
    save_final_population::Bool = false
end

Base.@kwdef struct RNGConfig
    seed::UInt64 = 0x6a09e667f3bcc909
    algorithm::Symbol = :xoshiro
end

Base.@kwdef struct PhysicsSettings
    phi_initial::Float64 = 0.0
    dphi_initial::Float64 = 0.0
    n_initial::Float64 = -21.0
    n_final::Float64 = 0.0
    shooting_lower::Float64 = 0.0
    shooting_upper::Float64 = 5.0
    shooting_tolerance::Float64 = 1e-6
    omega_gamma_h2::Float64 = 2.47739207701433e-5
    neff::Float64 = 3.044
    model_id::String = "nonminimal-scalar-tensor-v1"
end

Base.@kwdef struct RunConfig
    ga::GASettings = GASettings()
    seeding::SeedSettings = SeedSettings()
    minimizer::MinimizeSettings = MinimizeSettings()
    penalty::PenaltySettings = PenaltySettings()
    physics::PhysicsSettings = PhysicsSettings()
    rng::RNGConfig = RNGConfig()
    storage::StorageConfig = StorageConfig()
end

struct PenaltyBreakdown
    functional::Float64
    structural::Float64
    sign::Float64
    constants::Float64
    fisher::Float64
    total::Float64
end

PenaltyBreakdown(x::NamedTuple) = PenaltyBreakdown(
    x.functional, x.structural, x.sign, x.constants, x.fisher, x.total)

struct FittedCandidate
    individual::Individual
    param_names::Vector{Symbol}
    best_params::Vector{Float64}
    chi2::Float64
    description_length::Float64
    penalty::PenaltyBreakdown
    fisher_diags::Vector{Float64}
    fisher_status::Symbol
    converged::Bool
    accepted::Bool
    termination::Symbol
    objective_evals::Int
    optimizer_iterations::Int
    likelihood_names::Vector{Symbol}
    likelihood_chi2::Vector{Float64}
end

struct GenerationMetrics
    generation::Int
    ga_evaluations::Int
    ga_evaluations_generation::Int
    objective_evaluations::Int
    objective_evaluations_generation::Int
    elapsed_seconds::Float64
    generation_best_chi2::Float64
    generation_best_score::Float64
    generation_best_complexity::Int
    global_best_chi2::Float64
    global_best_score::Float64
    global_best_complexity::Int
    functional_penalty::Float64
    structural_penalty::Float64
    sign_penalty::Float64
    constants_penalty::Float64
    fisher_penalty::Float64
    accepted_count::Int
    rejected_count::Int
    unconverged_count::Int
    unique_candidate_count::Int
    score_median::Float64
    score_q10::Float64
    score_q90::Float64
    improvement::Float64
    generations_since_improvement::Int
    best_param_names::Vector{Symbol}
    best_params::Vector{Float64}
end

mutable struct GARunState
    completed_generation::Int
    population::Vector{FittedCandidate}
    global_best::FittedCandidate
    best_chi2::FittedCandidate
    historical_archive::Vector{FittedCandidate}
    history::Vector{GenerationMetrics}
    ga_evaluations::Int
    objective_evaluations::Int
    generations_since_improvement::Int
    elapsed_seconds::Float64
    rng::Random.Xoshiro
end

mutable struct HDF5RunStorage
    run_id::String
    run_dir::String
    config::RunConfig
    config_hash::String
    degraded::Bool
end

struct RunResult
    config::RunConfig
    metadata::Dict{String,Any}
    best_description_length::FittedCandidate
    best_chi2::FittedCandidate
    final_archive::Vector{FittedCandidate}
    historical_archive::Vector{FittedCandidate}
    history::Vector{GenerationMetrics}
end

struct EncodedTree
    node_kind::Vector{UInt8}
    operator_code::Vector{UInt16}
    feature_code::Vector{UInt16}
    constant_value::Vector{Float64}
end

function _encode_node!(encoded::EncodedTree, node::Node{Float64})
    if node.degree == 0
        if node.constant
            push!(encoded.node_kind, 0x01)
            push!(encoded.operator_code, 0x0000)
            push!(encoded.feature_code, 0x0000)
            push!(encoded.constant_value, node.val)
        else
            push!(encoded.node_kind, 0x02)
            push!(encoded.operator_code, 0x0000)
            push!(encoded.feature_code, node.feature)
            push!(encoded.constant_value, NaN)
        end
    elseif node.degree == 1
        node.op == 1 || throw(ArgumentError("unsupported unary operator index $(node.op)"))
        push!(encoded.node_kind, 0x03)
        push!(encoded.operator_code, 0x0065) # stable exp code 101
        push!(encoded.feature_code, 0x0000)
        push!(encoded.constant_value, NaN)
        _encode_node!(encoded, node.l)
    elseif node.degree == 2
        node.op in (1, 2, 3) || throw(ArgumentError("unsupported binary operator index $(node.op)"))
        push!(encoded.node_kind, 0x04)
        push!(encoded.operator_code, UInt16(node.op))
        push!(encoded.feature_code, 0x0000)
        push!(encoded.constant_value, NaN)
        _encode_node!(encoded, node.l)
        _encode_node!(encoded, node.r)
    else
        throw(ArgumentError("invalid tree degree $(node.degree)"))
    end
    return encoded
end

function encode_tree(tree::Node{Float64})
    encoded = EncodedTree(UInt8[], UInt16[], UInt16[], Float64[])
    return _encode_node!(encoded, tree)
end

function _decode_node(encoded::EncodedTree, index::Base.RefValue{Int})
    index[] <= length(encoded.node_kind) || throw(ArgumentError("truncated encoded tree"))
    i = index[]
    index[] += 1
    kind = encoded.node_kind[i]
    if kind == 0x01
        return Node(Float64; val=encoded.constant_value[i])
    elseif kind == 0x02
        encoded.feature_code[i] == 1 || throw(ArgumentError(
            "unsupported feature code $(encoded.feature_code[i])"))
        return Node(Float64; feature=Int(encoded.feature_code[i]))
    elseif kind == 0x03
        encoded.operator_code[i] == 101 || throw(ArgumentError("unsupported unary code"))
        return Node(1, _decode_node(encoded, index))
    elseif kind == 0x04
        op = encoded.operator_code[i]
        op in (1, 2, 3) || throw(ArgumentError("unsupported binary code $op"))
        left = _decode_node(encoded, index)
        right = _decode_node(encoded, index)
        return Node(Int(op), left, right)
    end
    throw(ArgumentError("invalid node kind $kind"))
end

function decode_tree(encoded::EncodedTree)
    lengths = (length(encoded.node_kind), length(encoded.operator_code),
        length(encoded.feature_code), length(encoded.constant_value))
    all(==(length(encoded.node_kind)), lengths) || throw(ArgumentError(
        "encoded tree arrays have inconsistent lengths"))
    index = Ref(1)
    tree = _decode_node(encoded, index)
    index[] == length(encoded.node_kind) + 1 || throw(ArgumentError("encoded tree has trailing nodes"))
    return tree
end

function encode_candidate(candidate::FittedCandidate)
    return (
        F=encode_tree(candidate.individual.F), U=encode_tree(candidate.individual.U),
        birth_generation=candidate.individual.birth_gen,
        complexity=candidate.individual.complexity,
        param_names=copy(candidate.param_names), best_params=copy(candidate.best_params),
        chi2=candidate.chi2, description_length=candidate.description_length,
        penalty=candidate.penalty, fisher_diags=copy(candidate.fisher_diags),
        fisher_status=candidate.fisher_status, converged=candidate.converged,
        accepted=candidate.accepted, termination=candidate.termination,
        objective_evals=candidate.objective_evals,
        optimizer_iterations=candidate.optimizer_iterations,
        likelihood_names=copy(candidate.likelihood_names),
        likelihood_chi2=copy(candidate.likelihood_chi2),
    )
end

function decode_candidate(saved)
    F = decode_tree(saved.F); U = decode_tree(saved.U)
    complexity = compute_complexity(F, U)
    complexity == saved.complexity || throw(ArgumentError("stored candidate complexity mismatch"))
    ind = Individual(F, U, saved.birth_generation, complexity)
    return FittedCandidate(ind, copy(saved.param_names), copy(saved.best_params), saved.chi2,
        saved.description_length, saved.penalty, copy(saved.fisher_diags), saved.fisher_status,
        saved.converged, saved.accepted, saved.termination, saved.objective_evals,
        saved.optimizer_iterations, copy(saved.likelihood_names), copy(saved.likelihood_chi2))
end

function copy_fitted(candidate::FittedCandidate)
    ind = Individual(copy_node(candidate.individual.F), copy_node(candidate.individual.U),
        candidate.individual.birth_gen, candidate.individual.complexity)
    return FittedCandidate(ind, copy(candidate.param_names), copy(candidate.best_params),
        candidate.chi2, candidate.description_length, candidate.penalty,
        copy(candidate.fisher_diags), candidate.fisher_status, candidate.converged,
        candidate.accepted, candidate.termination, candidate.objective_evals,
        candidate.optimizer_iterations, copy(candidate.likelihood_names),
        copy(candidate.likelihood_chi2))
end

function fitted_candidate(result, penalty_settings::PenaltySettings)
    parts = PenaltyBreakdown(penalty_components(result.individual, result.fisher_diags;
        fisher_status=result.fisher_status, settings=penalty_settings))
    return FittedCandidate(result.individual, Vector{Symbol}(result.param_names),
        Vector{Float64}(result.best_params), result.chi2_total, result.chi2_total + parts.total,
        parts, Vector{Float64}(result.fisher_diags), result.fisher_status,
        result.converged, result.accepted, result.termination, result.objective_evals,
        result.optimizer_iterations, Vector{Symbol}(result.likelihood_names),
        Vector{Float64}(result.likelihood_chi2))
end

function candidate_digest(candidate::FittedCandidate)
    io = IOBuffer()
    for tree in (candidate.individual.F, candidate.individual.U)
        encoded = encode_tree(tree)
        write(io, encoded.node_kind)
        write(io, encoded.operator_code)
        write(io, encoded.feature_code)
        write(io, reinterpret(UInt8, encoded.constant_value))
    end
    for name in candidate.param_names
        write(io, String(name)); write(io, UInt8(0))
    end
    write(io, reinterpret(UInt8, candidate.best_params))
    return bytes2hex(sha256(take!(io)))
end

function resolved_top_k(config::RunConfig)
    k = something(config.storage.top_k, max(1, ceil(Int, 0.10 * config.ga.Npops)))
    return clamp(k, 1, config.ga.Npops)
end

function update_archive!(archive::Vector{FittedCandidate}, candidates, k::Int)
    by_digest = Dict(candidate_digest(c) => copy_fitted(c) for c in archive)
    for candidate in candidates
        key = candidate_digest(candidate)
        if !haskey(by_digest, key) || candidate.description_length < by_digest[key].description_length
            by_digest[key] = copy_fitted(candidate)
        end
    end
    ordered = sort!(collect(values(by_digest)); by=c -> c.description_length)
    empty!(archive)
    append!(archive, ordered[1:min(k, length(ordered))])
    return archive
end

function final_archive(state::GARunState, config::RunConfig)
    order = sortperm(state.population; by=c -> c.description_length)
    count = min(resolved_top_k(config), length(order))
    return copy_fitted.(@view state.population[order[1:count]])
end

function _append_tree!(kind, op, feature, value, offsets, tree)
    encoded = encode_tree(tree)
    append!(kind, encoded.node_kind)
    append!(op, encoded.operator_code)
    append!(feature, encoded.feature_code)
    append!(value, encoded.constant_value)
    push!(offsets, length(kind) + 1)
end

function write_candidates(group, candidates::Vector{FittedCandidate})
    n = length(candidates)
    kind = UInt8[]; op = UInt16[]; feature = UInt16[]; value = Float64[]
    tree_offsets = Int64[1]
    parameter_names = String[]; parameter_values = Float64[]; parameter_offsets = Int64[1]
    fisher_values = Float64[]; fisher_offsets = Int64[1]
    likelihood_names = String[]; likelihood_values = Float64[]; likelihood_offsets = Int64[1]
    for candidate in candidates
        _append_tree!(kind, op, feature, value, tree_offsets, candidate.individual.F)
        _append_tree!(kind, op, feature, value, tree_offsets, candidate.individual.U)
        append!(parameter_names, String.(candidate.param_names))
        append!(parameter_values, candidate.best_params)
        push!(parameter_offsets, length(parameter_values) + 1)
        append!(fisher_values, candidate.fisher_diags)
        push!(fisher_offsets, length(fisher_values) + 1)
        append!(likelihood_names, String.(candidate.likelihood_names))
        append!(likelihood_values, candidate.likelihood_chi2)
        push!(likelihood_offsets, length(likelihood_values) + 1)
    end
    write(group, "count", Int64(n))
    write(group, "node_kind", kind); write(group, "operator_code", op)
    write(group, "feature_code", feature); write(group, "constant_value", value)
    write(group, "tree_offsets", tree_offsets)
    write(group, "birth_generation", Int64[c.individual.birth_gen for c in candidates])
    write(group, "complexity", Int64[c.individual.complexity for c in candidates])
    write(group, "parameter_names", parameter_names)
    write(group, "parameter_values", parameter_values)
    write(group, "parameter_offsets", parameter_offsets)
    write(group, "chi2", Float64[c.chi2 for c in candidates])
    write(group, "description_length", Float64[c.description_length for c in candidates])
    for field in (:functional, :structural, :sign, :constants, :fisher, :total)
        write(group, "penalty_$(field)", Float64[getfield(c.penalty, field) for c in candidates])
    end
    write(group, "fisher_values", fisher_values); write(group, "fisher_offsets", fisher_offsets)
    write(group, "fisher_status", String[String(c.fisher_status) for c in candidates])
    write(group, "converged", UInt8[c.converged for c in candidates])
    write(group, "accepted", UInt8[c.accepted for c in candidates])
    write(group, "termination", String[String(c.termination) for c in candidates])
    write(group, "objective_evals", Int64[c.objective_evals for c in candidates])
    write(group, "optimizer_iterations", Int64[c.optimizer_iterations for c in candidates])
    write(group, "likelihood_names", likelihood_names)
    write(group, "likelihood_chi2", likelihood_values)
    write(group, "likelihood_offsets", likelihood_offsets)
    return group
end

function _tree_slice(kind, operator_code, feature_code, constant_value, start::Int, stop::Int)
    return EncodedTree(
        kind[start:stop], operator_code[start:stop], feature_code[start:stop],
        constant_value[start:stop],
    )
end

function read_candidates(group)
    n = Int(read(group["count"]))
    n == 0 && return FittedCandidate[]
    node_kind = read(group["node_kind"])
    operator_code = read(group["operator_code"])
    feature_code = read(group["feature_code"])
    constant_value = read(group["constant_value"])
    tree_offsets = Int.(read(group["tree_offsets"]))
    param_offsets = Int.(read(group["parameter_offsets"]))
    fisher_offsets = Int.(read(group["fisher_offsets"]))
    likelihood_offsets = Int.(read(group["likelihood_offsets"]))
    param_names = read(group["parameter_names"])
    param_values = read(group["parameter_values"])
    fisher_values = read(group["fisher_values"])
    likelihood_names = read(group["likelihood_names"])
    likelihood_values = read(group["likelihood_chi2"])
    birth = read(group["birth_generation"]); complexity = read(group["complexity"])
    chi2 = read(group["chi2"]); score = read(group["description_length"])
    fisher_statuses = read(group["fisher_status"])
    converged = read(group["converged"]); accepted = read(group["accepted"])
    termination = read(group["termination"]); objective_evals = read(group["objective_evals"])
    optimizer_iterations = read(group["optimizer_iterations"])
    penalty_arrays = Dict(field => read(group["penalty_$(field)"])
        for field in (:functional, :structural, :sign, :constants, :fisher, :total))
    candidates = FittedCandidate[]
    for i in 1:n
        fstart, fstop = tree_offsets[2i - 1], tree_offsets[2i] - 1
        ustart, ustop = tree_offsets[2i], tree_offsets[2i + 1] - 1
        F = decode_tree(_tree_slice(node_kind, operator_code, feature_code, constant_value,
            fstart, fstop))
        U = decode_tree(_tree_slice(node_kind, operator_code, feature_code, constant_value,
            ustart, ustop))
        stored_complexity = Int(complexity[i])
        actual_complexity = compute_complexity(F, U)
        if actual_complexity != stored_complexity
            @warn "Stored candidate complexity does not match its trees; using the decoded tree complexity" candidate=i stored_complexity actual_complexity
        end
        ind = Individual(F, U, Int(birth[i]), actual_complexity)
        ps, pe = param_offsets[i], param_offsets[i + 1] - 1
        fs, fe = fisher_offsets[i], fisher_offsets[i + 1] - 1
        names = ps <= pe ? Symbol.(param_names[ps:pe]) : Symbol[]
        params = ps <= pe ? Vector{Float64}(param_values[ps:pe]) : Float64[]
        fisher = fs <= fe ? Vector{Float64}(fisher_values[fs:fe]) : Float64[]
        ls, le = likelihood_offsets[i], likelihood_offsets[i + 1] - 1
        like_names = ls <= le ? Symbol.(likelihood_names[ls:le]) : Symbol[]
        like_values = ls <= le ? Vector{Float64}(likelihood_values[ls:le]) : Float64[]
        parts = PenaltyBreakdown((getindex(penalty_arrays[field], i) for field in
            (:functional, :structural, :sign, :constants, :fisher, :total))...)
        push!(candidates, FittedCandidate(ind, names, params, chi2[i], score[i], parts,
            fisher, Symbol(fisher_statuses[i]), Bool(converged[i]), Bool(accepted[i]),
            Symbol(termination[i]), Int(objective_evals[i]), Int(optimizer_iterations[i]),
            like_names, like_values))
    end
    return candidates
end

function write_history(group, history::Vector{GenerationMetrics})
    write(group, "count", Int64(length(history)))
    scalar_fields = filter(f -> !(f in (:best_param_names, :best_params)), fieldnames(GenerationMetrics))
    for field in scalar_fields
        values = [getfield(row, field) for row in history]
        write(group, String(field), values)
    end
    names = String[]; values = Float64[]; offsets = Int64[1]
    for row in history
        append!(names, String.(row.best_param_names)); append!(values, row.best_params)
        push!(offsets, length(values) + 1)
    end
    write(group, "best_param_names", names); write(group, "best_params", values)
    write(group, "best_param_offsets", offsets)
end

function read_history(group)
    n = Int(read(group["count"]))
    n == 0 && return GenerationMetrics[]
    fields = filter(f -> !(f in (:best_param_names, :best_params)), fieldnames(GenerationMetrics))
    columns = Dict(field => read(group[String(field)]) for field in fields)
    names = read(group["best_param_names"]); values = read(group["best_params"])
    offsets = Int.(read(group["best_param_offsets"]))
    rows = GenerationMetrics[]
    for i in 1:n
        start, stop = offsets[i], offsets[i + 1] - 1
        pnames = start <= stop ? Symbol.(names[start:stop]) : Symbol[]
        params = start <= stop ? Vector{Float64}(values[start:stop]) : Float64[]
        args = (columns[field][i] for field in fields)
        push!(rows, GenerationMetrics(args..., pnames, params))
    end
    return rows
end

function config_dict(config::RunConfig)
    dictionary = Dict{String,Any}(
        "ga" => Dict("Npops" => config.ga.Npops, "Ngens" => config.ga.Ngens,
            "tour_size" => config.ga.tour_size, "selection_rate" => config.ga.selection_rate,
            "crossover_rate" => config.ga.crossover_rate, "mutation_rate" => config.ga.mutation_rate,
            "max_size" => config.ga.max_size, "likelihoods" => String.(config.ga.likelihoods)),
        "minimizer" => Dict(String(f) => getfield(config.minimizer, f)
            for f in fieldnames(MinimizeSettings)),
        "penalty" => Dict("mode" => String(config.penalty.mode),
            "fisher_pruning" => config.penalty.fisher_pruning),
        "physics" => Dict(String(f) => getfield(config.physics, f) for f in fieldnames(PhysicsSettings)),
        "rng" => Dict("seed" => string(config.rng.seed), "algorithm" => String(config.rng.algorithm)),
        "storage" => Dict("root" => config.storage.root,
            "top_k" => something(config.storage.top_k, -1),
            "save_final_population" => config.storage.save_final_population,
            "checkpoint_interval" => something(config.storage.checkpoint.interval_generations, -1),
            "checkpoint_retain" => config.storage.checkpoint.retain,
            "checkpoint_on_error" => String(config.storage.checkpoint.on_error)),
    )
    if !_is_default_seed_settings(config.seeding)
        dictionary["seeding"] = Dict{String,Any}(
            "guesses" => [Dict("label" => guess.label, "F" => guess.F, "U" => guess.U)
                for guess in config.seeding.guesses],
            "initial_fraction" => config.seeding.initial_fraction,
            "neighbor_mutations_min" => config.seeding.neighbor_mutations_min,
            "neighbor_mutations_max" => config.seeding.neighbor_mutations_max,
            "mutation_retry_limit" => config.seeding.mutation_retry_limit,
            "require_function_sanity" => config.seeding.require_function_sanity,
            "reject_duplicate_guesses" => config.seeding.reject_duplicate_guesses,
            "reinject_after" => something(config.seeding.reinject_after, -1),
            "reinject_interval" => config.seeding.reinject_interval,
            "reinjection_fraction" => config.seeding.reinjection_fraction,
        )
    end
    return dictionary
end

"""Return whether seeding uses its legacy-compatible default configuration.

Generated with GPT 5.6.
"""
function _is_default_seed_settings(settings::SeedSettings)
    defaults = SeedSettings()
    return isempty(settings.guesses) &&
        settings.initial_fraction == defaults.initial_fraction &&
        settings.neighbor_mutations_min == defaults.neighbor_mutations_min &&
        settings.neighbor_mutations_max == defaults.neighbor_mutations_max &&
        settings.mutation_retry_limit == defaults.mutation_retry_limit &&
        settings.require_function_sanity == defaults.require_function_sanity &&
        settings.reject_duplicate_guesses == defaults.reject_duplicate_guesses &&
        settings.reinject_after === defaults.reinject_after &&
        settings.reinject_interval == defaults.reinject_interval &&
        settings.reinjection_fraction == defaults.reinjection_fraction
end

function config_from_dict(d)
    ga = d["ga"]
    settings = GASettings(Npops=ga["Npops"], Ngens=ga["Ngens"], tour_size=ga["tour_size"],
        selection_rate=ga["selection_rate"], crossover_rate=ga["crossover_rate"],
        mutation_rate=ga["mutation_rate"], max_size=ga["max_size"],
        likelihoods=Symbol.(ga["likelihoods"]))
    m = d["minimizer"]
    minimizer = MinimizeSettings((m[String(f)] for f in fieldnames(MinimizeSettings))...)
    p = d["penalty"]
    penalty = PenaltySettings(mode=Symbol(p["mode"]), fisher_pruning=p["fisher_pruning"])
    ph = d["physics"]
    physics = PhysicsSettings(; (f => ph[String(f)] for f in fieldnames(PhysicsSettings))...)
    r = d["rng"]
    rng = RNGConfig(seed=parse(UInt64, r["seed"]), algorithm=Symbol(r["algorithm"]))
    s = d["storage"]
    interval = s["checkpoint_interval"] < 0 ? nothing : Int(s["checkpoint_interval"])
    top_k = s["top_k"] < 0 ? nothing : Int(s["top_k"])
    checkpoint = CheckpointConfig(interval_generations=interval, retain=s["checkpoint_retain"],
        on_error=Symbol(s["checkpoint_on_error"]))
    storage = StorageConfig(root=s["root"], checkpoint=checkpoint, top_k=top_k,
        save_final_population=s["save_final_population"])
    seeding = if haskey(d, "seeding")
        seed = d["seeding"]
        reinject_after = seed["reinject_after"] < 0 ? nothing : Int(seed["reinject_after"])
        SeedSettings(
            guesses=[GuessSpec(label=guess["label"], F=guess["F"], U=guess["U"])
                for guess in seed["guesses"]],
            initial_fraction=seed["initial_fraction"],
            neighbor_mutations_min=seed["neighbor_mutations_min"],
            neighbor_mutations_max=seed["neighbor_mutations_max"],
            mutation_retry_limit=seed["mutation_retry_limit"],
            require_function_sanity=seed["require_function_sanity"],
            reject_duplicate_guesses=seed["reject_duplicate_guesses"],
            reinject_after=reinject_after,
            reinject_interval=seed["reinject_interval"],
            reinjection_fraction=seed["reinjection_fraction"],
        )
    else
        SeedSettings()
    end
    return RunConfig(ga=settings, seeding=seeding, minimizer=minimizer, penalty=penalty,
        physics=physics, rng=rng, storage=storage)
end

config_fingerprint(config::RunConfig) = bytes2hex(sha256(codeunits(repr(config_dict(config)))))

function _file_sha256(path::String)
    return open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

function collect_run_metadata(config::RunConfig)
    root = dirname(@__DIR__)
    # A deployed copy is usually not a git repository: the defaults apply, and git's complaints stay out of the job log.
    safe_read(cmd, default="unknown") = try strip(read(pipeline(cmd; stderr=devnull), String)) catch; default end
    metadata = Dict{String,Any}(
        "julia_version" => string(VERSION), "architecture" => string(Sys.ARCH),
        "kernel" => string(Sys.KERNEL), "threads" => Threads.maxthreadid(),
        "git_commit" => safe_read(`git -C $root rev-parse HEAD`),
        "git_branch" => safe_read(`git -C $root branch --show-current`),
        "git_dirty" => !isempty(safe_read(`git -C $root status --short`, "")),
        "git_diff_sha256" => bytes2hex(sha256(codeunits(safe_read(`git -C $root diff HEAD`, "")))),
        "project_sha256" => _file_sha256(joinpath(root, "Project.toml")),
        "manifest_sha256" => _file_sha256(joinpath(root, "Manifest.toml")),
        "started_utc" => string(now(UTC)), "model_id" => config.physics.model_id,
        "units" => "H0=100h km s^-1 Mpc^-1; c in km s^-1; redshift dimensionless",
    )
    datasets = Dict{String,Any}()
    if isdefined(Main, :DATA_PATHS)
        for name in config.ga.likelihoods
            haskey(DATA_PATHS, name) || continue
            datasets[String(name)] = [Dict("path" => relpath(path, root),
                "size" => filesize(path), "sha256" => _file_sha256(path)) for path in DATA_PATHS[name]]
        end
    end
    metadata["datasets"] = datasets
    return metadata
end

function _atomic_toml(path::String, dictionary)
    mkpath(dirname(path))
    temp, io = mktemp(dirname(path); cleanup=false)
    try
        TOML.print(io, dictionary); flush(io); close(io)
        Base.Filesystem.rename(temp, path)
    catch
        isopen(io) && close(io)
        isfile(temp) && rm(temp; force=true)
        rethrow()
    end
end

function log_storage_event(storage::HDF5RunStorage, event::String; details="")
    open(joinpath(storage.run_dir, "events.jsonl"), "a") do io
        escaped = replace(String(details), '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n")
        println(io, "{\"time_utc\":\"$(now(UTC))\",\"event\":\"$event\",\"details\":\"$escaped\"}")
    end
end

function initialize_run_storage(config::RunConfig, metadata=collect_run_metadata(config))
    validate_run_config(config)
    run_id = string(uuid4())
    stamp = Dates.format(now(UTC), dateformat"yyyymmddTHHMMSSZ")
    run_dir = abspath(joinpath(config.storage.root, "$(stamp)_$(run_id)"))
    ispath(run_dir) && error("run directory already exists: $run_dir")
    mkpath(joinpath(run_dir, "checkpoints"))
    _atomic_toml(joinpath(run_dir, "config.toml"), config_dict(config))
    root = dirname(@__DIR__)
    cp(joinpath(root, "Project.toml"), joinpath(run_dir, "Project.toml"))
    cp(joinpath(root, "Manifest.toml"), joinpath(run_dir, "Manifest.toml"))
    metadata["run_id"] = run_id
    metadata["schema_major"] = RUN_SCHEMA_MAJOR
    metadata["schema_minor"] = RUN_SCHEMA_MINOR
    _atomic_toml(joinpath(run_dir, "metadata.toml"), metadata)
    storage = HDF5RunStorage(run_id, run_dir, config, config_fingerprint(config), false)
    log_storage_event(storage, "run_initialized")
    return storage
end

function validate_run_config(config::RunConfig)
    validate_selection_settings(config.ga)
    validate_seed_settings(config.seeding, config.ga)
    build_exact_seeds(config.seeding.guesses, 1, config.ga.max_size, config.seeding)
    validate_penalty_settings(config.penalty)
    config.rng.algorithm == :xoshiro || throw(ArgumentError("only :xoshiro RNG is supported"))
    interval = config.storage.checkpoint.interval_generations
    interval === nothing || interval > 0 || throw(ArgumentError("checkpoint interval must be positive"))
    config.storage.checkpoint.retain >= 1 || throw(ArgumentError("checkpoint retain must be at least 1"))
    config.storage.checkpoint.on_error in (:warn, :abort) || throw(ArgumentError(
        "checkpoint on_error must be :warn or :abort"))
    config.storage.top_k === nothing || config.storage.top_k > 0 || throw(ArgumentError("top_k must be positive"))
    config.penalty.mode == :all_coefficients && !config.penalty.fisher_pruning &&
        throw(ArgumentError(":all_coefficients currently requires fisher_pruning=true"))
    return nothing
end

function _write_root(file, storage, kind, state, status)
    attrs = HDF5.attributes(file)
    attrs["format_id"] = RUN_FORMAT_ID; attrs["file_kind"] = kind
    attrs["schema_major"] = RUN_SCHEMA_MAJOR; attrs["schema_minor"] = RUN_SCHEMA_MINOR
    attrs["run_id"] = storage.run_id; attrs["status"] = String(status)
    attrs["completed_generation"] = state.completed_generation
    attrs["config_sha256"] = storage.config_hash; attrs["created_utc"] = string(now(UTC))
    registry = create_group(file, "operator_registry")
    write(registry, "codes", UInt16[1, 2, 3, 101])
    write(registry, "names", ["add", "subtract", "multiply", "exp"])
    write(registry, "arities", UInt8[2, 2, 2, 1])
end

function _write_state(file, storage, state; kind, status, include_population=false)
    _write_root(file, storage, kind, state, status)
    write_candidates(create_group(file, "best_description_length"), [state.global_best])
    write_candidates(create_group(file, "best_chi2"), [state.best_chi2])
    write_candidates(create_group(file, "historical_archive"), state.historical_archive)
    write_candidates(create_group(file, "final_archive"), final_archive(state, storage.config))
    write_history(create_group(file, "history"), state.history)
    if include_population
        write_candidates(create_group(file, "population"), state.population)
        rng_group = create_group(file, "rng")
        write(rng_group, "algorithm", "xoshiro")
        write(rng_group, "state", UInt64[getfield(state.rng, f) for f in fieldnames(Random.Xoshiro)])
    end
    counters = create_group(file, "counters")
    write(counters, "ga_evaluations", Int64(state.ga_evaluations))
    write(counters, "objective_evaluations", Int64(state.objective_evaluations))
    write(counters, "generations_since_improvement", Int64(state.generations_since_improvement))
    write(counters, "elapsed_seconds", state.elapsed_seconds)
end

function _validate_h5(path::String, expected_kind::String)
    h5open(path, "r") do file
        attrs = HDF5.attributes(file)
        read(attrs["format_id"]) == RUN_FORMAT_ID || error("invalid run format")
        Int(read(attrs["schema_major"])) == RUN_SCHEMA_MAJOR || error("unsupported schema major")
        read(attrs["file_kind"]) == expected_kind || error("unexpected file kind")
        haskey(file, "best_description_length") || error("missing best candidate")
        haskey(file, "history") || error("missing history")
        expected_kind == "checkpoint" && !haskey(file, "population") && error("missing checkpoint population")
    end
    return true
end

function _atomic_h5(writer::Function, path::String, expected_kind::String)
    mkpath(dirname(path))
    temp, io = mktemp(dirname(path); cleanup=false); close(io)
    try
        h5open(temp, "w") do file
            writer(file)
        end
        _validate_h5(temp, expected_kind)
        digest = _file_sha256(temp)
        Base.Filesystem.rename(temp, path)
        _atomic_toml(path * ".sha256.toml", Dict("sha256" => digest))
        return path
    catch
        isfile(temp) && rm(temp; force=true)
        rethrow()
    end
end

function _verify_checksum(path::String)
    sidecar = path * ".sha256.toml"
    isfile(sidecar) || throw(ArgumentError("missing checkpoint checksum: $sidecar"))
    expected = TOML.parsefile(sidecar)["sha256"]
    _file_sha256(path) == expected || throw(ArgumentError("checksum mismatch for $path"))
end

function save_snapshot!(storage::HDF5RunStorage, state::GARunState; status=:running)
    path = joinpath(storage.run_dir, "progress.h5")
    _atomic_h5(path, "snapshot") do file
        _write_state(file, storage, state; kind="snapshot", status, include_population=false)
    end
    return path
end

function _apply_retention(storage::HDF5RunStorage)
    directory = joinpath(storage.run_dir, "checkpoints")
    files = sort(filter(path -> occursin(r"checkpoint-g\d+\.h5$", path), readdir(directory; join=true)))
    excess = length(files) - storage.config.storage.checkpoint.retain
    for path in files[1:max(0, excess)]
        rm(path; force=true); rm(path * ".sha256.toml"; force=true)
    end
end

function save_checkpoint!(storage::HDF5RunStorage, state::GARunState; final=false, status=:running)
    suffix = final ? "-final" : ""
    filename = "checkpoint-g$(lpad(state.completed_generation, 6, '0'))$(suffix).h5"
    path = joinpath(storage.run_dir, "checkpoints", filename)
    _atomic_h5(path, "checkpoint") do file
        _write_state(file, storage, state; kind="checkpoint", status, include_population=true)
    end
    _atomic_toml(joinpath(storage.run_dir, "checkpoints", "latest.toml"),
        Dict("path" => filename, "generation" => state.completed_generation,
            "sha256" => _file_sha256(path)))
    !final && _apply_retention(storage)
    log_storage_event(storage, "checkpoint_saved"; details=filename)
    return path
end

function _read_config_for_checkpoint(path::String)
    run_dir = dirname(dirname(abspath(path)))
    return config_from_dict(TOML.parsefile(joinpath(run_dir, "config.toml"))), run_dir
end

function validate_checkpoint(path::String, config::Union{Nothing,RunConfig}=nothing)
    _verify_checksum(path); _validate_h5(path, "checkpoint")
    h5open(path, "r") do file
        attrs = HDF5.attributes(file)
        if config !== nothing && read(attrs["config_sha256"]) != config_fingerprint(config)
            throw(ArgumentError("checkpoint configuration does not match requested configuration"))
        end
        population = read_candidates(file["population"])
        isempty(population) && throw(ArgumentError("checkpoint population is empty"))
        config !== nothing && length(population) != config.ga.Npops && throw(ArgumentError(
            "checkpoint population size does not match Npops"))
        state = read(file["rng/state"])
        length(state) == 5 || throw(ArgumentError("invalid Xoshiro RNG state"))
    end
    return true
end

function load_checkpoint(path::String)
    config, run_dir = _read_config_for_checkpoint(path)
    validate_checkpoint(path, config)
    return h5open(path, "r") do file
        population = read_candidates(file["population"])
        best = only(read_candidates(file["best_description_length"]))
        best_chi2 = only(read_candidates(file["best_chi2"]))
        archive = read_candidates(file["historical_archive"])
        history = read_history(file["history"])
        rng_state = read(file["rng/state"])
        rng = Random.Xoshiro(rng_state...)
        counters = file["counters"]
        state = GARunState(Int(read(HDF5.attributes(file)["completed_generation"])), population,
            best, best_chi2, archive, history, Int(read(counters["ga_evaluations"])),
            Int(read(counters["objective_evaluations"])),
            Int(read(counters["generations_since_improvement"])),
            Float64(read(counters["elapsed_seconds"])), rng)
        storage = HDF5RunStorage(String(read(HDF5.attributes(file)["run_id"])), run_dir, config,
            config_fingerprint(config), false)
        return (config=config, state=state, storage=storage)
    end
end

function finalize_run!(storage::HDF5RunStorage, state::GARunState; status=:completed)
    save_checkpoint!(storage, state; final=true, status)
    path = joinpath(storage.run_dir, "result.h5")
    _atomic_h5(path, "result") do file
        _write_state(file, storage, state; kind="result", status,
            include_population=storage.config.storage.save_final_population)
    end
    save_snapshot!(storage, state; status)
    metadata_path = joinpath(storage.run_dir, "metadata.toml")
    metadata = TOML.parsefile(metadata_path)
    metadata["status"] = String(status); metadata["finished_utc"] = string(now(UTC))
    metadata["completed_generation"] = state.completed_generation
    _atomic_toml(metadata_path, metadata)
    log_storage_event(storage, "run_finalized"; details=String(status))
    return path
end

function load_run_result(path::String)
    _verify_checksum(path); _validate_h5(path, "result")
    run_dir = dirname(abspath(path))
    config = config_from_dict(TOML.parsefile(joinpath(run_dir, "config.toml")))
    metadata = TOML.parsefile(joinpath(run_dir, "metadata.toml"))
    return h5open(path, "r") do file
        RunResult(config, metadata, only(read_candidates(file["best_description_length"])),
            only(read_candidates(file["best_chi2"])), read_candidates(file["final_archive"]),
            read_candidates(file["historical_archive"]), read_history(file["history"]))
    end
end

function best_candidate(result::RunResult; criterion=:description_length)
    criterion == :description_length && return result.best_description_length
    criterion == :chi2 && return result.best_chi2
    throw(ArgumentError("criterion must be :description_length or :chi2"))
end

generation_history(result::RunResult) = result.history

function record_generation!(storage::HDF5RunStorage, state::GARunState,
    metrics::GenerationMetrics)
    push!(state.history, metrics)
    log_storage_event(storage, "generation_recorded"; details=string(metrics.generation))
    return metrics
end

function reconstruct_functions(candidate::FittedCandidate)
    return build_F_funcs(copy_node(candidate.individual.F)), build_U_funcs(copy_node(candidate.individual.U))
end

function reconstruct_background(candidate::FittedCandidate; physics::PhysicsSettings=PhysicsSettings())
    params = Dict(zip(candidate.param_names, candidate.best_params))
    F_func, U_func = reconstruct_functions(candidate)
    p = CosmoParams(params[:Om0], 0.0, F_func, U_func)
    h = get(params, :h, 0.685); Ob0 = get(params, :Ob0, 0.0475)
    cache = setup_cosmo_cache(p, h, physics.dphi_initial)
    return Background(cache, p, h, Ob0, physics.dphi_initial)
end

reconstruct_background(candidate::FittedCandidate, config::RunConfig) =
    reconstruct_background(candidate; physics=config.physics)

function resume_run(path::String; overrides=nothing)
    loaded = load_checkpoint(path)
    overrides === nothing || throw(ArgumentError("resume overrides are not supported in schema v1"))
    return run_ga(loaded.config; resume_from=path)
end
