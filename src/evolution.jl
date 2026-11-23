using SciMLBase
using Roots
using OrdinaryDiffEqTsit5
using SciMLLogging
using StaticArrays


struct CosmoParams{TF,TU}
    Om0::Float64
    U0::Float64
    F_funcs::TF
    U_funcs::TU
end


@enum PhysicalStatus::UInt8 begin
    PHYSICS_OK
    PHYSICS_NONFINITE_STATE
    PHYSICS_NONFINITE_DENSITY
    PHYSICS_NONFINITE_FUNCTION
    PHYSICS_NONPOSITIVE_F
    PHYSICS_NONFINITE_DENOMINATOR
    PHYSICS_NONPOSITIVE_DENOMINATOR
    PHYSICS_NONFINITE_KINETIC_DETERMINANT
    PHYSICS_NONPOSITIVE_KINETIC_DETERMINANT
    PHYSICS_NONFINITE_NUMERATOR
    PHYSICS_NONPOSITIVE_NUMERATOR
    PHYSICS_NONFINITE_Q
    PHYSICS_NONPOSITIVE_Q
    PHYSICS_NONFINITE_DYNAMICS
end


struct FriedmannState
    status::PhysicalStatus
    q::Float64
    invq::Float64
    F::Float64
    F_Phi::Float64
    F_PhiPhi::Float64
    U_ga::Float64
    U_Phi::Float64
    U_total::Float64
    D::Float64
    K::Float64
    numerator::Float64
    rho_m::Float64
    rho_r::Float64
    a_inv3::Float64
end


struct PhysicalDomainError <: Exception
    status::PhysicalStatus
    N::Float64
    Phi::Float64
    X::Float64
end


function Base.showerror(io::IO, error::PhysicalDomainError)
    print(io, "physical-domain failure ", error.status,
        " at N=", error.N, ", Phi=", error.Phi, ", dPhi/dN=", error.X)
end


@inline function _friedmann_state(status, q, invq, F, F_Phi, F_PhiPhi,
                                   U_ga, U_Phi, U_total, D, K, numerator,
                                   rho_m, rho_r, a_inv3)
    return FriedmannState(
        status,
        Float64(q),
        Float64(invq),
        Float64(F),
        Float64(F_Phi),
        Float64(F_PhiPhi),
        Float64(U_ga),
        Float64(U_Phi),
        Float64(U_total),
        Float64(D),
        Float64(K),
        Float64(numerator),
        Float64(rho_m),
        Float64(rho_r),
        Float64(a_inv3),
    )
end


"""
Evaluate the algebraic Friedmann constraint without clipping invalid values.

The branch continuously connected to the early-time FLRW solution requires
`F > 0`, `D > 0`, `K > 0`, and a positive Friedmann numerator. In particular,
negative or zero `q = E^2` is reported explicitly and is never passed to
`sqrt` or `log`.
"""
@inline function inspect_friedmann(p, Or0, N, Phi, X)
    nan = NaN
    if !(isfinite(N) && isfinite(Phi) && isfinite(X) &&
         isfinite(Or0) && isfinite(p.Om0) && isfinite(p.U0))
        return _friedmann_state(PHYSICS_NONFINITE_STATE, nan, nan, nan, nan,
            nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    end

    a_inv = exp(-N)
    a_inv3 = a_inv * a_inv * a_inv
    a_inv4 = a_inv3 * a_inv
    rho_m = p.Om0 * a_inv3
    rho_r = Or0 * a_inv4
    if !(isfinite(a_inv3) && isfinite(rho_m) && isfinite(rho_r))
        return _friedmann_state(PHYSICS_NONFINITE_DENSITY, nan, nan, nan, nan,
            nan, nan, nan, nan, nan, nan, nan, rho_m, rho_r, a_inv3)
    end

    F, F_Phi, F_PhiPhi = p.F_funcs(Phi)
    U_ga, U_Phi = p.U_funcs(Phi)
    U_total = p.U0 + U_ga
    if !(isfinite(F) && isfinite(F_Phi) && isfinite(F_PhiPhi) &&
         isfinite(U_ga) && isfinite(U_Phi) && isfinite(U_total))
        return _friedmann_state(PHYSICS_NONFINITE_FUNCTION, nan, nan, F,
            F_Phi, F_PhiPhi, U_ga, U_Phi, U_total, nan, nan, nan,
            rho_m, rho_r, a_inv3)
    end
    X2 = X * X
    D = 3.0 * F - 0.5 * X2 + 3.0 * F_Phi * X
    K = 2.0 * F + 3.0 * F_Phi * F_Phi
    numerator = 3.0 * (rho_m + rho_r) + U_total

    if F <= 0.0
        return _friedmann_state(PHYSICS_NONPOSITIVE_F, nan, nan, F, F_Phi,
            F_PhiPhi, U_ga, U_Phi, U_total, D, K, numerator,
            rho_m, rho_r, a_inv3)
    elseif !isfinite(D)
        return _friedmann_state(PHYSICS_NONFINITE_DENOMINATOR, nan, nan, F,
            F_Phi, F_PhiPhi, U_ga, U_Phi, U_total, D, K, numerator,
            rho_m, rho_r, a_inv3)
    elseif D <= 0.0
        return _friedmann_state(PHYSICS_NONPOSITIVE_DENOMINATOR, nan, nan, F,
            F_Phi, F_PhiPhi, U_ga, U_Phi, U_total, D, K, numerator,
            rho_m, rho_r, a_inv3)
    elseif !isfinite(K)
        return _friedmann_state(PHYSICS_NONFINITE_KINETIC_DETERMINANT, nan,
            nan, F, F_Phi, F_PhiPhi, U_ga, U_Phi, U_total, D, K,
            numerator, rho_m, rho_r, a_inv3)
    elseif K <= 0.0
        return _friedmann_state(PHYSICS_NONPOSITIVE_KINETIC_DETERMINANT, nan,
            nan, F, F_Phi, F_PhiPhi, U_ga, U_Phi, U_total, D, K,
            numerator, rho_m, rho_r, a_inv3)
    elseif !isfinite(numerator)
        return _friedmann_state(PHYSICS_NONFINITE_NUMERATOR, nan, nan, F,
            F_Phi, F_PhiPhi, U_ga, U_Phi, U_total, D, K, numerator,
            rho_m, rho_r, a_inv3)
    elseif numerator <= 0.0
        return _friedmann_state(PHYSICS_NONPOSITIVE_NUMERATOR, nan, nan, F,
            F_Phi, F_PhiPhi, U_ga, U_Phi, U_total, D, K, numerator,
            rho_m, rho_r, a_inv3)
    end

    q = numerator / D
    invq = D / numerator
    if !(isfinite(q) && isfinite(invq))
        return _friedmann_state(PHYSICS_NONFINITE_Q, q, invq, F, F_Phi,
            F_PhiPhi, U_ga, U_Phi, U_total, D, K, numerator,
            rho_m, rho_r, a_inv3)
    elseif q <= 0.0 || invq <= 0.0
        return _friedmann_state(PHYSICS_NONPOSITIVE_Q, q, invq, F, F_Phi,
            F_PhiPhi, U_ga, U_Phi, U_total, D, K, numerator,
            rho_m, rho_r, a_inv3)
    end

    return _friedmann_state(PHYSICS_OK, q, invq, F, F_Phi, F_PhiPhi,
        U_ga, U_Phi, U_total, D, K, numerator, rho_m, rho_r, a_inv3)
end


@inline function friedmann_quantities(p, Or0, N, Phi, X)
    result = inspect_friedmann(p, Or0, N, Phi, X)
    result.status == PHYSICS_OK ||
        throw(PhysicalDomainError(result.status, Float64(N), Float64(Phi), Float64(X)))
    return result
end


"""
Solve directly for `s = d(log H)/dN` and `Xprime = dX/dN`.

This avoids forming a large `qprime` and dividing it by `q` later. The
determinant of the two-equation system is `K = 2F + 3F_Phi^2`.
"""
@inline function compute_field_dynamics(p, Or0, N, Phi, X)
    fr = friedmann_quantities(p, Or0, N, Phi, X)

    X2 = X * X
    rho_plus_p_over_H2 = (3.0 * fr.rho_m + 4.0 * fr.rho_r) * fr.invq
    U_Phi_over_H2 = fr.U_Phi * fr.invq

    b_H = -rho_plus_p_over_H2 - (1.0 + fr.F_PhiPhi) * X2 + fr.F_Phi * X
    b_Phi = -3.0 * X - U_Phi_over_H2 + 6.0 * fr.F_Phi

    s = (b_H - fr.F_Phi * b_Phi) / fr.K
    Xprime = ((2.0 * fr.F + fr.F_Phi * X) * b_Phi -
              (X - 3.0 * fr.F_Phi) * b_H) / fr.K

    if !(isfinite(s) && isfinite(Xprime))
        throw(PhysicalDomainError(
            PHYSICS_NONFINITE_DYNAMICS, Float64(N), Float64(Phi), Float64(X)))
    end

    qprime_over_q = 2.0 * s
    q_prime = fr.q * qprime_over_q
    return (
        Xprime=Xprime,
        ddPhi=Xprime,
        ddΦ=Xprime,
        s=s,
        dlnH_dN=s,
        q=fr.q,
        invq=fr.invq,
        q_prime=q_prime,
        qprime_over_q=qprime_over_q,
        F=fr.F,
        F_Phi=fr.F_Phi,
        F_Φ=fr.F_Phi,
        F_PhiPhi=fr.F_PhiPhi,
        F_ΦΦ=fr.F_PhiPhi,
        D=fr.D,
        K=fr.K,
        numerator=fr.numerator,
        rho_m=fr.rho_m,
        rho_r=fr.rho_r,
        a_inv3=fr.a_inv3,
    )
end


mutable struct EvolutionContext{TP}
    cosmo::TP
    Or0::Float64
    last_physical_status::PhysicalStatus
    stage_physical_status::PhysicalStatus
    saw_invalid_stage::Bool
    domain_rejections::Int
end


@inline function _mark_invalid_stage!(context::EvolutionContext, status)
    context.saw_invalid_stage = true
    context.stage_physical_status = status
    return nothing
end


"""
Evaluate a tentative Runge--Kutta stage without aborting Tsit5.

For finite algebraic quantities the field equations have a real continuation
through negative `D/P`. This continuation is used only to let the adaptive
solver finish assessing an oversized tentative step. `outside_physical_domain`
then rejects every step which touched a nonphysical stage, so these values can
never enter an accepted solution.
"""
@inline function compute_field_dynamics_stage!(context::EvolutionContext, N, Phi, X)
    fr = inspect_friedmann(context.cosmo, context.Or0, N, Phi, X)
    if fr.status != PHYSICS_OK
        _mark_invalid_stage!(context, fr.status)
    end

    if !(isfinite(X) && isfinite(fr.F) && isfinite(fr.F_Phi) &&
         isfinite(fr.F_PhiPhi) && isfinite(fr.U_Phi) &&
         isfinite(fr.D) && isfinite(fr.K) && fr.K != 0.0 &&
         isfinite(fr.numerator) && fr.numerator != 0.0 &&
         isfinite(fr.rho_m) && isfinite(fr.rho_r))
        _mark_invalid_stage!(context,
            fr.status == PHYSICS_OK ? PHYSICS_NONFINITE_DYNAMICS : fr.status)
        # Finite placeholders for a step guaranteed to be rejected below.
        return (Xprime=0.0, s=0.0, invq=1.0)
    end

    invq = fr.status == PHYSICS_OK ? fr.invq : fr.D / fr.numerator
    X2 = X * X
    rho_plus_p_over_H2 = (3.0 * fr.rho_m + 4.0 * fr.rho_r) * invq
    U_Phi_over_H2 = fr.U_Phi * invq
    b_H = -rho_plus_p_over_H2 - (1.0 + fr.F_PhiPhi) * X2 + fr.F_Phi * X
    b_Phi = -3.0 * X - U_Phi_over_H2 + 6.0 * fr.F_Phi
    s = (b_H - fr.F_Phi * b_Phi) / fr.K
    Xprime = ((2.0 * fr.F + fr.F_Phi * X) * b_Phi -
              (X - 3.0 * fr.F_Phi) * b_H) / fr.K

    if !(isfinite(invq) && isfinite(s) && isfinite(Xprime))
        _mark_invalid_stage!(context, PHYSICS_NONFINITE_DYNAMICS)
        return (Xprime=0.0, s=0.0, invq=1.0)
    end
    return (Xprime=Xprime, s=s, invq=invq)
end


function evolution_ode_field(u, context::EvolutionContext, N)
    X = u[2]
    result = compute_field_dynamics_stage!(context, N, u[1], X)
    dPhi = isfinite(X) ? X : 0.0
    return @SVector [dPhi, result.Xprime]
end


function outside_physical_domain(u, context::EvolutionContext, N)
    Phi = u[1]
    X = u[2]
    result = inspect_friedmann(context.cosmo, context.Or0, N, Phi, X)
    endpoint_invalid = result.status != PHYSICS_OK
    stage_invalid = context.saw_invalid_stage
    if endpoint_invalid || stage_invalid
        context.last_physical_status = endpoint_invalid ?
            result.status : context.stage_physical_status
        context.domain_rejections += 1
    end
    # OrdinaryDiffEq calls `isoutofdomain` once after every attempted adaptive
    # step and before acceptance, including steps rejected by the error test.
    context.saw_invalid_stage = false
    context.stage_physical_status = PHYSICS_OK
    return endpoint_invalid || stage_invalid
end


function setup_shooting_integrator(p_base::CosmoParams, Or0::Float64, X_ini=0.0;
                                   Phi_ini=0.0, N_ini=-21.0,
                                   reltol=1e-8, abstol=1e-10,
                                   maxiters=10_000)
    y0 = @SVector [Float64(Phi_ini), Float64(X_ini)]
    context = EvolutionContext(
        p_base, Or0, PHYSICS_OK, PHYSICS_OK, false, 0)
    problem = ODEProblem(evolution_ode_field, y0, (Float64(N_ini), 0.0), context)

    return init(
        problem,
        Tsit5(),
        save_everystep=false,
        save_start=false,
        save_end=false,
        reltol=reltol,
        abstol=abstol,
        maxiters=maxiters,
        isoutofdomain=outside_physical_domain,
        unstable_check=(dt, u, p, t) -> !all(isfinite, u),
        verbose=SciMLLogging.None(),
    )
end


@enum ShootingTrialStatus::UInt8 begin
    SHOOTING_TRIAL_SUCCESS
    SHOOTING_TRIAL_INVALID_INPUT
    SHOOTING_TRIAL_PHYSICAL_FAILURE
    SHOOTING_TRIAL_ODE_FAILURE
    SHOOTING_TRIAL_NONFINITE_RESIDUAL
end


struct ShootingTrial
    status::ShootingTrialStatus
    physical_status::PhysicalStatus
    U0::Float64
    residual::Float64
    q0::Float64
    Phi0::Float64
    X0::Float64
    last_N::Float64
    retcode::SciMLBase.ReturnCode.T
end


shooting_success(trial::ShootingTrial) =
    trial.status == SHOOTING_TRIAL_SUCCESS && isfinite(trial.residual)


struct InvalidShootingTrialError <: Exception
    trial::ShootingTrial
end


function Base.showerror(io::IO, error::InvalidShootingTrialError)
    trial = error.trial
    print(io, "invalid shooting trial at U0=", trial.U0,
        " (status=", trial.status, ", physical_status=", trial.physical_status,
        ", retcode=", trial.retcode, ", last_N=", trial.last_N, ")")
end


function _failed_trial(status, physical_status, U0, integrator;
                       q0=NaN, Phi0=NaN, X0=NaN, residual=NaN)
    return ShootingTrial(
        status,
        physical_status,
        Float64(U0),
        Float64(residual),
        Float64(q0),
        Float64(Phi0),
        Float64(X0),
        Float64(integrator.t),
        integrator.sol.retcode,
    )
end


"""Evaluate one physical `U0` value without rejecting the whole candidate."""
function evaluate_shooting_trial!(U0_val, integrator, p_base::CosmoParams,
                                  Or0::Float64, u0)
    if !(isfinite(U0_val) && isfinite(Or0) && all(isfinite, u0))
        return _failed_trial(SHOOTING_TRIAL_INVALID_INPUT,
            PHYSICS_NONFINITE_STATE, U0_val, integrator)
    end

    p_new = CosmoParams(p_base.Om0, Float64(U0_val), p_base.F_funcs, p_base.U_funcs)
    context = integrator.p
    context isa EvolutionContext ||
        throw(ArgumentError("shooting integrator does not use an EvolutionContext"))
    context.cosmo = p_new
    context.Or0 = Or0
    context.last_physical_status = PHYSICS_OK
    context.stage_physical_status = PHYSICS_OK
    context.saw_invalid_stage = false
    context.domain_rejections = 0

    reinit!(integrator, u0; erase_sol=true, reset_dt=true)
    try
        solve!(integrator)
    catch error
        error isa InterruptException && rethrow()
        if error isa PhysicalDomainError
            return _failed_trial(SHOOTING_TRIAL_PHYSICAL_FAILURE,
                error.status, U0_val, integrator)
        end
        rethrow()
    end

    if !SciMLBase.successful_retcode(integrator.sol)
        physical_status = context.last_physical_status
        status = physical_status == PHYSICS_OK ?
            SHOOTING_TRIAL_ODE_FAILURE : SHOOTING_TRIAL_PHYSICAL_FAILURE
        return _failed_trial(status, physical_status, U0_val, integrator)
    end

    Phi0 = integrator.u[1]
    X0 = integrator.u[2]
    fr = inspect_friedmann(p_new, Or0, 0.0, Phi0, X0)
    if fr.status != PHYSICS_OK
        return _failed_trial(SHOOTING_TRIAL_PHYSICAL_FAILURE, fr.status,
            U0_val, integrator; q0=fr.q, Phi0=Phi0, X0=X0)
    end

    # `inspect_friedmann` has already established q0 > 0. This explicit guard
    # prevents a future change from ever forwarding a non-positive q to log.
    if !(isfinite(fr.q) && fr.q > 0.0)
        physical_status = isfinite(fr.q) ? PHYSICS_NONPOSITIVE_Q : PHYSICS_NONFINITE_Q
        return _failed_trial(SHOOTING_TRIAL_PHYSICAL_FAILURE, physical_status,
            U0_val, integrator; q0=fr.q, Phi0=Phi0, X0=X0)
    end

    residual = log(fr.q)
    if !isfinite(residual)
        return _failed_trial(SHOOTING_TRIAL_NONFINITE_RESIDUAL,
            PHYSICS_NONFINITE_Q, U0_val, integrator;
            q0=fr.q, Phi0=Phi0, X0=X0)
    end

    return ShootingTrial(
        SHOOTING_TRIAL_SUCCESS,
        PHYSICS_OK,
        Float64(U0_val),
        residual,
        fr.q,
        Float64(Phi0),
        Float64(X0),
        Float64(integrator.t),
        integrator.sol.retcode,
    )
end


function shooting_U0(U0_val, integrator, p_base::CosmoParams, Or0::Float64)
    trial = evaluate_shooting_trial!(
        U0_val, integrator, p_base, Or0, integrator.sol.prob.u0)
    shooting_success(trial) || throw(InvalidShootingTrialError(trial))
    return trial.residual
end


Base.@kwdef struct ShootingSettings
    normalization_bounds::Tuple{Float64,Float64} = (-10_000.0, 10_000.0)
    initial_scale::Float64 = 0.5
    expansion_factor::Float64 = 2.0
    max_expansions::Int = 15
    fallback_scan_points::Int = 33
    validation_depth::Int = 2
    xatol::Float64 = 1e-8
    closure_atol::Float64 = 1e-7
    root_maxiters::Int = 100
end


@enum ShootingRootStatus::UInt8 begin
    SHOOTING_ROOT_SUCCESS
    SHOOTING_ROOT_INVALID_INPUT
    SHOOTING_ROOT_NO_PHYSICAL_TRIAL
    SHOOTING_ROOT_NO_BRACKET
    SHOOTING_ROOT_REFINEMENT_FAILED
end


struct ShootingRootResult
    status::ShootingRootStatus
    U0::Float64
    normalization::Float64
    residual::Float64
    bracket::Tuple{Float64,Float64}
    trial_count::Int
    last_trial::ShootingTrial
end


shooting_success(result::ShootingRootResult) =
    result.status == SHOOTING_ROOT_SUCCESS && isfinite(result.U0)


struct ShootingSolveError <: Exception
    result::ShootingRootResult
end


function Base.showerror(io::IO, error::ShootingSolveError)
    result = error.result
    print(io, "U0 shooting failed (status=", result.status,
        ", trials=", result.trial_count,
        ", last_trial_status=", result.last_trial.status,
        ", last_physical_status=", result.last_trial.physical_status, ")")
end


struct ShootingObjective{TI,TP,TU}
    integrator::TI
    p_base::TP
    Or0::Float64
    u0::TU
    potential_offset::Float64
end


mutable struct ShootingSearch{TO}
    objective::TO
    normalizations::Vector{Float64}
    trials::Vector{ShootingTrial}
end


@inline physical_U0(objective::ShootingObjective, normalization) =
    normalization - objective.potential_offset


function _trial_at!(search::ShootingSearch, normalization)
    x = Float64(normalization)
    existing = findfirst(y -> isequal(y, x), search.normalizations)
    existing === nothing || return search.trials[existing]

    objective = search.objective
    trial = evaluate_shooting_trial!(
        physical_U0(objective, x),
        objective.integrator,
        objective.p_base,
        objective.Or0,
        objective.u0,
    )
    push!(search.normalizations, x)
    push!(search.trials, trial)
    return trial
end


function _residual_at!(search::ShootingSearch, normalization)
    trial = _trial_at!(search, normalization)
    shooting_success(trial) || throw(InvalidShootingTrialError(trial))
    return trial.residual
end


"""
Return exact roots and sign-changing brackets, resetting continuity whenever a
sample is invalid. This prevents false brackets across failed ODE trials.
"""
function find_contiguous_sign_brackets(xs, residuals, valid;
                                       residual_atol=1e-7)
    length(xs) == length(residuals) == length(valid) ||
        throw(DimensionMismatch("xs, residuals, and valid must have equal length"))
    order = sortperm(xs)
    exact = Float64[]
    brackets = Tuple{Float64,Float64}[]
    previous = 0

    for index in order
        if valid[index] && isfinite(residuals[index])
            x = Float64(xs[index])
            residual = Float64(residuals[index])
            if abs(residual) <= residual_atol
                push!(exact, x)
            elseif previous != 0
                previous_residual = Float64(residuals[previous])
                if signbit(residual) != signbit(previous_residual)
                    push!(brackets, (Float64(xs[previous]), x))
                end
            end
            previous = index
        else
            previous = 0
        end
    end
    return exact, brackets
end


function _available_brackets(search, closure_atol)
    residuals = getproperty.(search.trials, :residual)
    valid = shooting_success.(search.trials)
    return find_contiguous_sign_brackets(
        search.normalizations, residuals, valid; residual_atol=closure_atol)
end


function _validated_bracket!(search, bracket, settings)
    left, right = bracket
    left_trial = _trial_at!(search, left)
    right_trial = _trial_at!(search, right)
    shooting_success(left_trial) && shooting_success(right_trial) || return nothing

    for _ in 1:settings.validation_depth
        middle = left + 0.5 * (right - left)
        middle_trial = _trial_at!(search, middle)
        shooting_success(middle_trial) || return nothing
        if abs(middle_trial.residual) <= settings.closure_atol
            return (kind=:exact, x=middle, bracket=(middle, middle))
        elseif signbit(left_trial.residual) != signbit(middle_trial.residual)
            right = middle
            right_trial = middle_trial
        else
            left = middle
            left_trial = middle_trial
        end
    end
    return (kind=:bracket, x=NaN, bracket=(left, right))
end


function _successful_root(search, normalization, bracket, settings)
    trial = _trial_at!(search, normalization)
    if shooting_success(trial) && abs(trial.residual) <= settings.closure_atol
        return ShootingRootResult(
            SHOOTING_ROOT_SUCCESS,
            trial.U0,
            Float64(normalization),
            trial.residual,
            bracket,
            length(search.trials),
            trial,
        )
    end
    return nothing
end


function _try_available_roots!(search, guess, settings, attempted)
    exact, brackets = _available_brackets(search, settings.closure_atol)
    sort!(exact; by=x -> abs(x - guess))
    for x in exact
        result = _successful_root(search, x, (x, x), settings)
        result === nothing || return result
    end

    sort!(brackets; by=bracket -> abs(0.5 * (bracket[1] + bracket[2]) - guess))
    for raw_bracket in brackets
        raw_bracket in attempted && continue
        push!(attempted, raw_bracket)
        validated = _validated_bracket!(search, raw_bracket, settings)
        validated === nothing && continue
        if validated.kind == :exact
            result = _successful_root(
                search, validated.x, validated.bracket, settings)
            result === nothing || return result
            continue
        end

        bracket = validated.bracket
        try
            root = Roots.find_zero(
                x -> _residual_at!(search, x),
                bracket,
                Roots.Brent();
                xatol=settings.xatol,
                maxiters=settings.root_maxiters,
            )
            result = _successful_root(search, root, bracket, settings)
            result === nothing || return result
        catch error
            error isa InterruptException && rethrow()
            if !(error isa InvalidShootingTrialError ||
                 error isa Roots.ConvergenceFailed)
                rethrow()
            end
        end
    end
    return nothing
end


function _invalid_root_result(status, search)
    last_trial = isempty(search.trials) ? ShootingTrial(
        SHOOTING_TRIAL_INVALID_INPUT,
        PHYSICS_NONFINITE_STATE,
        NaN, NaN, NaN, NaN, NaN, NaN,
        ReturnCode.Default,
    ) : last(search.trials)
    return ShootingRootResult(
        status, NaN, NaN, NaN, (NaN, NaN), length(search.trials), last_trial)
end


function _validate_shooting_settings(settings::ShootingSettings)
    lower, upper = settings.normalization_bounds
    isfinite(lower) && isfinite(upper) && lower < upper ||
        throw(ArgumentError("normalization_bounds must be finite and ordered"))
    settings.initial_scale > 0.0 && isfinite(settings.initial_scale) ||
        throw(ArgumentError("initial_scale must be finite and positive"))
    settings.expansion_factor > 1.0 && isfinite(settings.expansion_factor) ||
        throw(ArgumentError("expansion_factor must be finite and greater than one"))
    settings.max_expansions >= 0 || throw(ArgumentError("max_expansions must be non-negative"))
    settings.fallback_scan_points >= 0 ||
        throw(ArgumentError("fallback_scan_points must be non-negative"))
    settings.validation_depth >= 0 ||
        throw(ArgumentError("validation_depth must be non-negative"))
    settings.xatol > 0.0 && settings.closure_atol > 0.0 ||
        throw(ArgumentError("shooting tolerances must be positive"))
    return settings
end


"""
Find the physical additive potential normalization with an adaptive bracket.

The search coordinate is

    normalization = U0 + U_ga(Phi_ref),

so constant shifts of the GA potential shift the returned `U0` but leave the
search and the total potential invariant.
"""
function find_U0!(integrator::SciMLBase.DEIntegrator,
                  p_base::CosmoParams,
                  Or0::Float64;
                  Phi_ini=0.0,
                  X_ini=nothing,
                  settings=ShootingSettings())
    _validate_shooting_settings(settings)
    default_u0 = integrator.sol.prob.u0
    Xi = isnothing(X_ini) ? default_u0[2] : X_ini
    u0 = @SVector [Float64(Phi_ini), Float64(Xi)]

    U_reference, _ = p_base.U_funcs(Phi_ini)
    F_reference, F_Phi_reference, _ = p_base.F_funcs(Phi_ini)
    D_reference = 3.0 * F_reference - 0.5 * Xi^2 +
                  3.0 * F_Phi_reference * Xi
    normalization_guess = D_reference - 3.0 * (p_base.Om0 + Or0)

    objective = ShootingObjective(
        integrator, p_base, Or0, u0, Float64(U_reference))
    search = ShootingSearch(objective, Float64[], ShootingTrial[])
    if !(isfinite(U_reference) && isfinite(D_reference) &&
         isfinite(normalization_guess) && all(isfinite, u0))
        return _invalid_root_result(SHOOTING_ROOT_INVALID_INPUT, search)
    end

    lower, upper = settings.normalization_bounds
    center = clamp(Float64(normalization_guess), lower, upper)
    attempted = Set{Tuple{Float64,Float64}}()

    _trial_at!(search, center)
    result = _try_available_roots!(search, center, settings, attempted)
    result === nothing || return result

    radius = settings.initial_scale
    for _ in 1:settings.max_expansions
        _trial_at!(search, max(lower, center - radius))
        _trial_at!(search, min(upper, center + radius))
        result = _try_available_roots!(search, center, settings, attempted)
        result === nothing || return result
        radius *= settings.expansion_factor
        center - radius <= lower && center + radius >= upper && break
    end

    # This is intentionally a fallback: successful candidates normally finish
    # from the estimate or geometric expansion after only a few ODE solves.
    if settings.fallback_scan_points >= 2
        for normalization in range(lower, upper; length=settings.fallback_scan_points)
            _trial_at!(search, normalization)
        end
        result = _try_available_roots!(search, center, settings, attempted)
        result === nothing || return result
    end

    any(shooting_success, search.trials) ||
        return _invalid_root_result(SHOOTING_ROOT_NO_PHYSICAL_TRIAL, search)
    _, brackets = _available_brackets(search, settings.closure_atol)
    status = isempty(brackets) ?
        SHOOTING_ROOT_NO_BRACKET : SHOOTING_ROOT_REFINEMENT_FAILED
    return _invalid_root_result(status, search)
end


function get_U0!(integrator::SciMLBase.DEIntegrator,
                 p_base::CosmoParams,
                 Or0::Float64;
                 Phi_ini=0.0,
                 X_ini=nothing,
                 settings=ShootingSettings())
    result = find_U0!(integrator, p_base, Or0;
        Phi_ini=Phi_ini, X_ini=X_ini, settings=settings)
    shooting_success(result) || throw(ShootingSolveError(result))
    return result.U0
end


# Compatibility with the positional call used by the original background code.
function get_U0!(integrator::SciMLBase.DEIntegrator,
                 p_base::CosmoParams,
                 Or0::Float64,
                 search_bounds,
                 xtol;
                 Phi_ini=0.0,
                 X_ini=nothing,
                 nscan=nothing)
    fallback_scan_points = isnothing(nscan) ? 33 : Int(nscan)
    settings = ShootingSettings(
        normalization_bounds=(Float64(search_bounds[1]), Float64(search_bounds[2])),
        xatol=Float64(xtol),
        fallback_scan_points=fallback_scan_points,
    )
    return get_U0!(integrator, p_base, Or0;
        Phi_ini=Phi_ini, X_ini=X_ini, settings=settings)
end


function evolution_ode_full(u, context::EvolutionContext, N)
    X = u[2]
    result = compute_field_dynamics_stage!(context, N, u[1], X)
    dPhi = isfinite(X) ? X : 0.0
    # Eta is triangular: it does not feed back into Phi or X. A negative invq
    # can occur only in a stage already marked for rejection, so this finite
    # continuation cannot affect an accepted physical trajectory.
    d_eta_dN = exp(-N) * sqrt(abs(result.invq))
    isfinite(d_eta_dN) || (d_eta_dN = 0.0)
    return @SVector [dPhi, result.Xprime, d_eta_dN]
end
