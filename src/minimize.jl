using SciMLBase
using FiniteDifferences
using OrdinaryDiffEqTsit5
using StaticArrays
using LinearAlgebra
using Optim
using Random
using SHA

const _MINIMIZE_DIR = @__DIR__
const _PROJECT_ROOT = dirname(_MINIMIZE_DIR)

include(joinpath(_MINIMIZE_DIR, "evolution.jl"))
include(joinpath(_MINIMIZE_DIR, "background.jl"))
include(joinpath(_MINIMIZE_DIR, "likelihoods.jl"))

const _RECOVERY_SCALES = (0.05, 0.15, 0.40, 0.80)

struct MinimizeSettings
    time_limit::Float64
    g_tol::Float64
    max_prune_iterations::Int
    fisher_fd_factor::Float64
    fisher_negative_tol::Float64
    invalid_objective::Float64
    rejection_objective::Float64
    recovery_enabled::Bool
    recovery_max_probes::Int
    recovery_time_fraction::Float64
end

MinimizeSettings() = MinimizeSettings(
    2.0, 1e-3, 8, 1e-3, 1e-4, 1e9, 1e6, true, 8, 0.20)

struct ParameterSpec
    param_names::Vector{Symbol}
    x_cosmo0::Vector{Float64}
    num_cosmo::Int
    num_F::Int
    num_U::Int
end

num_coefficients(spec::ParameterSpec) = spec.num_F + spec.num_U

struct ObjectiveEvaluation
    value::Float64
    status::Symbol
    diagnostic::String
    likelihood_chi2::Vector{Float64}
end

evaluation_valid(evaluation::ObjectiveEvaluation) =
    evaluation.status == :valid && isfinite(evaluation.value)

@inline function failure_rank(status::Symbol)
    status == :likelihood_invalid && return 1
    status == :full_background_failed && return 2
    status == :shooting_refinement_failed && return 3
    status == :shooting_no_bracket && return 4
    status == :shooting_no_physical_trial && return 5
    status == :function_invalid && return 6
    return 7
end

mutable struct FitResult
    x_opt::Vector{Float64}
    objective_value::Float64
    converged::Bool
    accepted::Bool
    termination::Symbol
    objective_evals::Int
    optimizer_iterations::Int
    likelihood_names::Vector{Symbol}
    likelihood_chi2::Vector{Float64}
    initial_status::Symbol
    final_status::Symbol
    recovery_probes::Int
    recovery_succeeded::Bool
    feasibility_evals::Int
end

struct MinimizationContext{TL,TC}
    likelihoods::TL
    cache::TC
end

mutable struct JointCosmoObjective{TL,TC,TF,TU}
    likelihoods::TL
    param_names::Vector{Symbol}
    cache::TC
    F_tree::TF
    U_tree::TU
    num_cosmo::Int
    num_F::Int
    num_U::Int
    invalid_objective::Float64
    objective_calls::Base.RefValue{Int}
    feasibility_calls::Base.RefValue{Int}
    likelihood_names::Vector{Symbol}
    last_likelihood_chi2::Vector{Float64}
    last_status::Base.RefValue{Symbol}
    last_diagnostic::Base.RefValue{String}
end

const _PARAMETER_BOUNDS = Dict{Symbol,Tuple{Float64,Float64}}(
    :Om0 => (0.2, 0.4),
    :h => (0.6, 0.8),
    :Ob0 => (0.02, 0.08),
    :sigma8_0 => (0.6, 1.0),
    :M_nuisance => (-20.0, -19.0),
)

@inline parameter_bounds(name::Symbol) = get(_PARAMETER_BOUNDS, name, (-Inf, Inf))

function _set_tree_coefficients!(obj::JointCosmoObjective, x::AbstractVector{Float64})
    if obj.num_F > 0
        first = obj.num_cosmo + 1
        set_optimizable_constants!(obj.F_tree, @view x[first:(first + obj.num_F - 1)])
    end
    if obj.num_U > 0
        first = obj.num_cosmo + obj.num_F + 1
        set_optimizable_constants!(obj.U_tree, @view x[first:(first + obj.num_U - 1)])
    end
    return nothing
end

function _parameter_values(obj::JointCosmoObjective, x::AbstractVector{Float64})
    values = Dict(zip(obj.param_names, @view x[1:obj.num_cosmo]))
    return (
        Om0=values[:Om0],
        h=values[:h],
        Ob0=get(values, :Ob0, 0.0475),
        sigma8=get(values, :sigma8_0, 0.800),
        M=get(values, :M_nuisance, -19.4),
    )
end

function _parameters_in_bounds(obj::JointCosmoObjective, x::AbstractVector{Float64})
    for (i, name) in enumerate(obj.param_names)
        lo, hi = parameter_bounds(name)
        isfinite(x[i]) && lo <= x[i] <= hi || return false
    end
    return true
end

function _shooting_failure_status(cache)
    result = cache.last_shooting_result[]
    result === nothing && return :shooting_no_physical_trial
    result.status == SHOOTING_ROOT_NO_PHYSICAL_TRIAL && return :shooting_no_physical_trial
    result.status == SHOOTING_ROOT_NO_BRACKET && return :shooting_no_bracket
    result.status == SHOOTING_ROOT_REFINEMENT_FAILED && return :shooting_refinement_failed
    return :shooting_no_physical_trial
end

function _background_failure_status(error::BackgroundSolveError, cache)
    error.status == BACKGROUND_SHOOTING_FAILURE && return _shooting_failure_status(cache)
    error.status in (BACKGROUND_FULL_PHYSICAL_FAILURE, BACKGROUND_FULL_ODE_FAILURE,
        BACKGROUND_CLOSURE_FAILURE, BACKGROUND_NONFINITE_SOLUTION) &&
        return :full_background_failed
    return :full_background_failed
end

function _invalid_evaluation(obj::JointCosmoObjective, status::Symbol, diagnostic::String)
    obj.last_status[] = status
    obj.last_diagnostic[] = diagnostic
    return ObjectiveEvaluation(obj.invalid_objective, status, diagnostic,
        fill(NaN, length(obj.likelihoods)))
end

"""Evaluate physical feasibility and, optionally, the complete likelihood."""
function evaluate_objective!(obj::JointCosmoObjective, x::AbstractVector{Float64};
    include_likelihood::Bool=true)
    include_likelihood ? (obj.objective_calls[] += 1) : (obj.feasibility_calls[] += 1)
    expected_length = obj.num_cosmo + obj.num_F + obj.num_U
    length(x) == expected_length || throw(DimensionMismatch(
        "expected $expected_length parameters, received $(length(x))"))
    all(isfinite, x) || return _invalid_evaluation(obj, :function_invalid,
        "parameter vector contains a non-finite value")
    _parameters_in_bounds(obj, x) || return _invalid_evaluation(obj, :function_invalid,
        "cosmological parameter lies outside its prior")

    _set_tree_coefficients!(obj, x)
    values = _parameter_values(obj, x)
    local bg
    try
        F_funcs = build_F_funcs(obj.F_tree)
        U_funcs = build_U_funcs(obj.U_tree)
        p_test = CosmoParams(values.Om0, 0.0, F_funcs, U_funcs)
        Phi_initial = find_initial_phi(p_test)
        F_initial, _, _ = F_funcs(Phi_initial)
        U_initial, _ = U_funcs(Phi_initial)
        if !(isfinite(F_initial) && F_initial > 0.0 && isfinite(U_initial))
            return _invalid_evaluation(obj, :function_invalid,
                "F and U must be finite and F must be positive at the resolved initial field")
        end
        bg = Background(
            obj.cache, p_test, values.h, values.Ob0, 0.0; Phi_ini=Phi_initial)
    catch error
        error isa InterruptException && rethrow()
        if error isa BackgroundSolveError
            return _invalid_evaluation(obj, _background_failure_status(error, obj.cache),
                sprint(showerror, error))
        elseif error isa InitialPhiSolveError || error isa PhysicalDomainError ||
               error isa ShootingSolveError ||
               error isa InvalidShootingTrialError || error isa DomainError ||
               error isa OverflowError
            return _invalid_evaluation(obj, :function_invalid, sprint(showerror, error))
        end
        rethrow()
    end

    SciMLBase.successful_retcode(bg.sol) || return _invalid_evaluation(
        obj, :full_background_failed, "background integrator returned $(bg.sol.retcode)")
    state_today = bg.sol(0.0)
    all(isfinite, state_today) || return _invalid_evaluation(
        obj, :full_background_failed, "present-day state is non-finite")

    if !include_likelihood
        obj.last_status[] = :valid
        obj.last_diagnostic[] = "background is physical"
        return ObjectiveEvaluation(0.0, :valid, "background is physical", Float64[])
    end

    components = similar(obj.last_likelihood_chi2)
    chi2_total = 0.0
    try
        for (i, like) in enumerate(obj.likelihoods)
            component = if like isa SH0ESLikelihood
                -2.0 * log_likelihood(like, bg, values.M)
            elseif like isa GrowthRateLikelihood
                -2.0 * log_likelihood(like, bg, values.sigma8)
            else
                -2.0 * log_likelihood(like, bg)
            end
            isfinite(component) || return _invalid_evaluation(obj, :likelihood_invalid,
                "$(obj.likelihood_names[i]) returned a non-finite chi-square")
            components[i] = component
            chi2_total += component
        end
    catch error
        error isa InterruptException && rethrow()
        if error isa PhysicalDomainError || error isa DomainError || error isa OverflowError ||
           error isa LinearAlgebra.PosDefException || error isa LinearAlgebra.SingularException
            return _invalid_evaluation(obj, :likelihood_invalid, sprint(showerror, error))
        end
        rethrow()
    end

    isfinite(chi2_total) || return _invalid_evaluation(
        obj, :likelihood_invalid, "total chi-square is non-finite")
    copyto!(obj.last_likelihood_chi2, components)
    obj.last_status[] = :valid
    obj.last_diagnostic[] = "valid"
    return ObjectiveEvaluation(chi2_total / 2.0, :valid, "valid", components)
end

function (obj::JointCosmoObjective)(x::AbstractVector{Float64})
    evaluation = evaluate_objective!(obj, x; include_likelihood=true)
    return evaluation_valid(evaluation) ? evaluation.value : obj.invalid_objective
end

build_minimization_context(likelihoods::Vector{AbstractLikelihood}, cosmo_cache) =
    MinimizationContext(likelihoods, cosmo_cache)

function build_parameter_spec(ind::Individual, likelihoods::Vector{AbstractLikelihood})
    param_names = Symbol[:Om0, :h]
    x_cosmo0 = Float64[0.295, 0.685]
    if any(l -> l isa SH0ESLikelihood, likelihoods)
        push!(param_names, :M_nuisance); push!(x_cosmo0, -19.4)
    end
    if any(l -> l isa GrowthRateLikelihood, likelihoods)
        push!(param_names, :sigma8_0); push!(x_cosmo0, 0.80)
    end
    if any(l -> l isa PlanckCompressed_Zhai2018 || l isa DESI_BAO, likelihoods)
        push!(param_names, :Ob0); push!(x_cosmo0, 0.0475)
    end
    num_F = length(get_optimizable_constants(ind.F))
    num_U = length(get_optimizable_constants(ind.U))
    return ParameterSpec(param_names, x_cosmo0, length(x_cosmo0), num_F, num_U)
end

function initial_joint_vector(spec::ParameterSpec, ind::Individual)
    return Vector{Float64}(vcat(spec.x_cosmo0,
        get_optimizable_constants(ind.F), get_optimizable_constants(ind.U)))
end

function build_joint_objective(context::MinimizationContext, spec::ParameterSpec,
    ind::Individual, settings::MinimizeSettings)
    return JointCosmoObjective(
        context.likelihoods, spec.param_names, context.cache, ind.F, ind.U,
        spec.num_cosmo, spec.num_F, spec.num_U, settings.invalid_objective,
        Ref(0), Ref(0), Symbol[nameof(typeof(like)) for like in context.likelihoods],
        zeros(length(context.likelihoods)), Ref(:not_evaluated), Ref(""))
end

"""Cheap local function check; the trajectory performs the authoritative checks."""
function passes_function_sanity(ind::Individual)
    try
        F_funcs = build_F_funcs(ind.F)
        U_funcs = build_U_funcs(ind.U)
        p = CosmoParams(0.295, 0.0, F_funcs, U_funcs)
        Phi_ini = find_initial_phi(p)
        F, _, _ = F_funcs(Phi_ini)
        U, _ = U_funcs(Phi_ini)
        return isfinite(F) && F > 0.0 && isfinite(U)
    catch error
        error isa InterruptException && rethrow()
        if error isa InitialPhiSolveError || error isa DomainError ||
           error isa OverflowError
            return false
        end
        rethrow()
    end
end

@inline function _reflect_to_interval(value::Float64, lo::Float64, hi::Float64)
    isfinite(lo) && isfinite(hi) || return value
    width = hi - lo
    width > 0.0 || return lo
    phase = mod(value - lo, 2.0 * width)
    return phase <= width ? lo + phase : hi - (phase - width)
end

function _stable_recovery_rng(ind::Individual)
    signature = individual_signature(ind)
    digest = sha256(codeunits(string(signature[1], '\0', signature[2])))
    seed = zero(UInt64)
    for byte in @view digest[1:8]
        seed = (seed << 8) | UInt64(byte)
    end
    return Random.Xoshiro(seed)
end

function recovery_probes(ind::Individual, spec::ParameterSpec, x0::Vector{Float64},
    max_probes::Int)
    rng = _stable_recovery_rng(ind)
    probes = Vector{Vector{Float64}}()
    sizehint!(probes, Base.min(max_probes, 2 * length(_RECOVERY_SCALES)))
    for scale in _RECOVERY_SCALES
        direction = ifelse.(rand(rng, Bool, length(x0)), 1.0, -1.0)
        for sign in (1.0, -1.0)
            probe = copy(x0)
            for i in eachindex(probe)
                step = if i <= spec.num_cosmo
                    lo, hi = parameter_bounds(spec.param_names[i])
                    hi - lo
                else
                    Base.max(abs(x0[i]), 1.0)
                end
                probe[i] += sign * scale * step * direction[i]
                if i <= spec.num_cosmo
                    lo, hi = parameter_bounds(spec.param_names[i])
                    probe[i] = _reflect_to_interval(probe[i], lo, hi)
                end
            end
            push!(probes, probe)
            length(probes) >= max_probes && return probes
        end
    end
    return probes
end

struct ScaledSimplexer <: Optim.Simplexer
    spec::ParameterSpec
end

function Optim.simplexer(simplexer::ScaledSimplexer, x0::Tx) where {Tx}
    spec = simplexer.spec
    simplex = Tx[copy(x0) for _ in 1:(length(x0) + 1)]
    for i in eachindex(x0)
        step = if i <= spec.num_cosmo
            lo, hi = parameter_bounds(spec.param_names[i])
            0.02 * (hi - lo)
        else
            0.05 * Base.max(abs(x0[i]), 1.0)
        end
        simplex[i + 1][i] += step
        if i <= spec.num_cosmo
            lo, hi = parameter_bounds(spec.param_names[i])
            simplex[i + 1][i] = _reflect_to_interval(simplex[i + 1][i], lo, hi)
        end
    end
    return simplex
end

@inline _repair_termination(status::Symbol) = Symbol("repair_exhausted_", status)

function _fit_result(obj, x, evaluation, converged, accepted, termination,
    iterations, initial_status, recovery_probes, recovery_succeeded)
    return FitResult(copy(x), evaluation.value, converged, accepted, termination,
        obj.objective_calls[], iterations, copy(obj.likelihood_names),
        copy(evaluation.likelihood_chi2), initial_status, evaluation.status,
        recovery_probes, recovery_succeeded, obj.feasibility_calls[])
end

function fit_once!(obj::JointCosmoObjective, spec::ParameterSpec, ind::Individual,
    x0::Vector{Float64}, settings::MinimizeSettings)
    started = time()
    deadline = started + settings.time_limit
    recovery_deadline = Base.min(
        deadline, started + settings.recovery_time_fraction * settings.time_limit)
    baseline = evaluate_objective!(obj, x0; include_likelihood=true)
    initial_status = baseline.status
    best_failure = baseline
    seed = copy(x0)
    seed_evaluation = baseline
    attempted = 0
    recovered = false

    if !evaluation_valid(baseline)
        if settings.recovery_enabled
            probes = recovery_probes(ind, spec, x0, settings.recovery_max_probes)
            for pair_start in 1:2:length(probes)
                time() < recovery_deadline || break
                valid_pair = Tuple{Vector{Float64},ObjectiveEvaluation}[]
                for index in pair_start:Base.min(pair_start + 1, length(probes))
                    time() < recovery_deadline || break
                    probe = probes[index]
                    attempted += 1
                    physical = evaluate_objective!(obj, probe; include_likelihood=false)
                    if !evaluation_valid(physical)
                        failure_rank(physical.status) < failure_rank(best_failure.status) &&
                            (best_failure = physical)
                        continue
                    end
                    complete = evaluate_objective!(obj, probe; include_likelihood=true)
                    if evaluation_valid(complete)
                        push!(valid_pair, (probe, complete))
                    elseif failure_rank(complete.status) < failure_rank(best_failure.status)
                        best_failure = complete
                    end
                end
                if !isempty(valid_pair)
                    best = argmin(last(item).value for item in valid_pair)
                    seed, seed_evaluation = valid_pair[best]
                    recovered = true
                    break
                end
            end
        end
        if !recovered
            _set_tree_coefficients!(obj, x0)
            return _fit_result(obj, x0, best_failure, false, false,
                _repair_termination(best_failure.status), 0, initial_status, attempted, false)
        end
    end

    _set_tree_coefficients!(obj, seed)
    remaining = deadline - time()
    if remaining <= 0.0
        termination = recovered ? :recovered_time_limit_before_optimizer :
            :time_limit_before_optimizer
        return _fit_result(obj, seed, seed_evaluation, false, true, termination,
            0, initial_status, attempted, recovered)
    end

    result = Optim.optimize(obj, seed,
        NelderMead(initial_simplex=ScaledSimplexer(spec)),
        Optim.Options(time_limit=remaining, g_tol=settings.g_tol))
    x_opt = Vector{Float64}(Optim.minimizer(result))
    final = evaluate_objective!(obj, x_opt; include_likelihood=true)
    converged = Optim.converged(result)
    accepted = all(isfinite, x_opt) && evaluation_valid(final) &&
        final.value < settings.rejection_objective
    termination = if accepted
        if recovered
            converged ? :recovered_converged : :recovered_time_or_iteration_limit
        else
            converged ? :converged : :time_or_iteration_limit
        end
    else
        Symbol("optimizer_ended_", final.status)
    end
    return _fit_result(obj, x_opt, final, converged, accepted, termination,
        Optim.iterations(result), initial_status, attempted, recovered)
end

function lock_best_fit!(ind::Individual, spec::ParameterSpec, x_opt::AbstractVector{Float64})
    if spec.num_F > 0
        first = spec.num_cosmo + 1
        set_optimizable_constants!(ind.F, @view x_opt[first:(first + spec.num_F - 1)])
    end
    if spec.num_U > 0
        first = spec.num_cosmo + spec.num_F + 1
        set_optimizable_constants!(ind.U, @view x_opt[first:(first + spec.num_U - 1)])
    end
    return ind
end

function compute_fisher_diagonal(obj::JointCosmoObjective, spec::ParameterSpec,
    fit::FitResult, settings::MinimizeSettings)
    n = num_coefficients(spec)
    n == 0 && return Float64[]
    diagonals = zeros(n)
    fdm = central_fdm(5, 2, factor=settings.fisher_fd_factor)
    for i in 1:n
        index = spec.num_cosmo + i
        diagonals[i] = fdm(fit.x_opt[index]) do value
            test = copy(fit.x_opt); test[index] = value; obj(test)
        end
    end
    return diagonals
end

function sanitize_fisher_diagonal(diagonals::Vector{Float64}, settings::MinimizeSettings)
    sanitized = copy(diagonals)
    for i in eachindex(sanitized)
        if !isfinite(sanitized[i])
            sanitized[i] = NaN
        elseif sanitized[i] < 0.0 && abs(sanitized[i]) < settings.fisher_negative_tol
            sanitized[i] = abs(sanitized[i])
        end
    end
    return sanitized
end

function fisher_precisions(diagonals::Vector{Float64})
    return [isfinite(value) && value > 0.0 ? sqrt(12.0 / value) : Inf
        for value in diagonals]
end

function coefficients_from_fit(spec::ParameterSpec, x_opt::AbstractVector{Float64})
    n = num_coefficients(spec)
    n == 0 && return Float64[]
    return Vector{Float64}(x_opt[(spec.num_cosmo + 1):(spec.num_cosmo + n)])
end

function fisher_status(spec::ParameterSpec, diagonals::AbstractVector{Float64})
    num_coefficients(spec) == 0 && return :not_applicable
    length(diagonals) == num_coefficients(spec) || return :fit_rejected
    all(isfinite, diagonals) || return :fit_rejected
    return :valid
end

function coefficients_to_zero(spec::ParameterSpec, fit::FitResult,
    diagonals::Vector{Float64})
    coefficients = coefficients_from_fit(spec, fit.x_opt)
    precisions = fisher_precisions(diagonals)
    return [i for i in eachindex(coefficients)
        if isfinite(precisions[i]) && abs(coefficients[i]) < precisions[i]]
end

function zero_coefficients!(ind::Individual, spec::ParameterSpec, fit::FitResult,
    indices::Vector{Int})
    isempty(indices) && return ind
    coefficients = coefficients_from_fit(spec, fit.x_opt)
    coefficients[indices] .= 0.0
    spec.num_F > 0 && set_optimizable_constants!(ind.F, coefficients[1:spec.num_F])
    spec.num_U > 0 && set_optimizable_constants!(
        ind.U, coefficients[(spec.num_F + 1):end])
    return ind
end

function simplify_after_pruning(ind::Individual)
    simplified = simplify_individual(ind)
    return Individual(simplified.F, simplified.U, ind.birth_gen,
        compute_complexity(simplified.F, simplified.U))
end

function make_result(ind::Individual, spec::ParameterSpec, fit::FitResult,
    diagonals::Vector{Float64}; status::Union{Nothing,Symbol}=nothing)
    best_cosmo = Vector{Float64}(fit.x_opt[1:spec.num_cosmo])
    return (
        individual=ind, param_names=spec.param_names, best_params=best_cosmo,
        chi2_total=fit.objective_value * 2.0,
        best_dict=Dict(zip(spec.param_names, best_cosmo)), fisher_diags=diagonals,
        fisher_status=status === nothing ? fisher_status(spec, diagonals) : status,
        converged=fit.converged, accepted=fit.accepted, termination=fit.termination,
        objective_evals=fit.objective_evals, optimizer_iterations=fit.optimizer_iterations,
        likelihood_names=fit.likelihood_names, likelihood_chi2=fit.likelihood_chi2,
        initial_status=fit.initial_status, final_status=fit.final_status,
        recovery_probes=fit.recovery_probes, recovery_succeeded=fit.recovery_succeeded,
        feasibility_evals=fit.feasibility_evals,
    )
end

function rejected_result(ind::Individual, likelihoods::Vector{AbstractLikelihood},
    settings::MinimizeSettings; termination=:function_invalid)
    spec = build_parameter_spec(ind, likelihoods)
    diagonals = num_coefficients(spec) == 0 ? Float64[] : fill(NaN, num_coefficients(spec))
    return (
        individual=ind, param_names=spec.param_names, best_params=copy(spec.x_cosmo0),
        chi2_total=settings.invalid_objective * 2.0,
        best_dict=Dict(zip(spec.param_names, spec.x_cosmo0)), fisher_diags=diagonals,
        fisher_status=num_coefficients(spec) == 0 ? :not_applicable : :fit_rejected,
        converged=false, accepted=false, termination, objective_evals=0,
        optimizer_iterations=0,
        likelihood_names=Symbol[nameof(typeof(like)) for like in likelihoods],
        likelihood_chi2=fill(NaN, length(likelihoods)), initial_status=:function_invalid,
        final_status=:function_invalid, recovery_probes=0, recovery_succeeded=false,
        feasibility_evals=0,
    )
end

function _fit_current!(current::Individual, context::MinimizationContext,
    settings::MinimizeSettings)
    spec = build_parameter_spec(current, context.likelihoods)
    x0 = initial_joint_vector(spec, current)
    objective = build_joint_objective(context, spec, current, settings)
    return spec, objective, fit_once!(objective, spec, current, x0, settings)
end

function iterative_fit_prune!(ind::Individual, context::MinimizationContext,
    settings::MinimizeSettings)
    current = ind
    last_result = nothing
    for _ in 1:settings.max_prune_iterations
        spec, objective, fit = _fit_current!(current, context, settings)
        if !fit.accepted
            diagonals = num_coefficients(spec) == 0 ? Float64[] : fill(NaN, num_coefficients(spec))
            return make_result(current, spec, fit, diagonals; status=
                num_coefficients(spec) == 0 ? :not_applicable : :fit_rejected)
        end
        lock_best_fit!(current, spec, fit.x_opt)
        diagonals = sanitize_fisher_diagonal(
            compute_fisher_diagonal(objective, spec, fit, settings), settings)
        fit.objective_evals = objective.objective_calls[]
        fit.feasibility_evals = objective.feasibility_calls[]
        lock_best_fit!(current, spec, fit.x_opt)
        if any(value -> value < 0.0, diagonals) && !fit.converged
            return make_result(current, spec, fit, diagonals; status=:invalid_negative)
        end
        last_result = make_result(current, spec, fit, diagonals)
        indices = coefficients_to_zero(spec, fit, diagonals)
        isempty(indices) && return last_result
        zero_coefficients!(current, spec, fit, indices)
        current = simplify_after_pruning(current)
    end
    return last_result
end

function iterative_fit_no_fisher!(ind::Individual, context::MinimizationContext,
    settings::MinimizeSettings)
    spec, _, fit = _fit_current!(ind, context, settings)
    if !fit.accepted
        diagonals = num_coefficients(spec) == 0 ? Float64[] : fill(NaN, num_coefficients(spec))
        return make_result(ind, spec, fit, diagonals; status=
            num_coefficients(spec) == 0 ? :not_applicable : :fit_rejected)
    end
    lock_best_fit!(ind, spec, fit.x_opt)
    return make_result(ind, spec, fit, Float64[]; status=:disabled)
end

function optimize_individual!(ind::Individual, likelihoods::Vector{AbstractLikelihood},
    cosmo_cache; settings::MinimizeSettings=MinimizeSettings(),
    penalty_settings::PenaltySettings=PenaltySettings())
    context = build_minimization_context(likelihoods, cosmo_cache)
    result = penalty_settings.mode == :all_coefficients && penalty_settings.fisher_pruning ?
        iterative_fit_prune!(ind, context, settings) :
        iterative_fit_no_fisher!(ind, context, settings)
    return result === nothing ? rejected_result(ind, likelihoods, settings;
        termination=:fit_pipeline_empty) : result
end
