using SciMLBase
using OrdinaryDiffEqTsit5
using ForwardDiff
using SciMLLogging
using QuadGK: quadgk, alloc_segbuf


const c_kms = 299792.458


struct CosmoCache{TS,TF,TG,TSeg,TR}
    shooting_integrator::TS
    full_integrator::TF
    growth_integrator::TG
    segbuf::TSeg
    last_shooting_result::TR
end


struct Background{Tsol,Tp,Tg,TSeg}
    sol::Tsol
    p::Tp
    H0::Float64
    ωb::Float64
    ωm::Float64
    ωγ::Float64
    Or0::Float64
    growth_integrator::Tg
    η_today::Float64
    segbuf::TSeg
end


# ASCII aliases are convenient in diagnostics while the actual fields retain
# the production Background API.
function Base.getproperty(bg::Background, name::Symbol)
    name === :omega_b && return getfield(bg, :ωb)
    name === :omega_m && return getfield(bg, :ωm)
    name === :omega_gamma && return getfield(bg, :ωγ)
    name === :eta_today && return getfield(bg, :η_today)
    return getfield(bg, name)
end


function Base.propertynames(bg::Background, private::Bool=false)
    fields = fieldnames(typeof(bg))
    aliases = (:omega_b, :omega_m, :omega_gamma, :eta_today)
    return (fields..., aliases...)
end


mutable struct GrowthContext{TB}
    background::Union{Nothing,TB}
end


@enum BackgroundFailureStatus::UInt8 begin
    BACKGROUND_SHOOTING_FAILURE
    BACKGROUND_FULL_PHYSICAL_FAILURE
    BACKGROUND_FULL_ODE_FAILURE
    BACKGROUND_CLOSURE_FAILURE
    BACKGROUND_NONFINITE_SOLUTION
end


struct BackgroundSolveError <: Exception
    status::BackgroundFailureStatus
    physical_status::PhysicalStatus
    retcode::SciMLBase.ReturnCode.T
    detail::String
end


function Base.showerror(io::IO, error::BackgroundSolveError)
    print(io, "background solve failed (status=", error.status,
        ", physical_status=", error.physical_status,
        ", retcode=", error.retcode, "): ", error.detail)
end


@inline function effective_gravity(F, F_Phi)
    K = 2.0 * F + 3.0 * F_Phi * F_Phi
    if !(isfinite(F) && isfinite(F_Phi) && isfinite(K) && F > 0.0 && K > 0.0)
        throw(PhysicalDomainError(
            F <= 0.0 ? PHYSICS_NONPOSITIVE_F :
            PHYSICS_NONPOSITIVE_KINETIC_DETERMINANT,
            NaN, NaN, NaN))
    end
    return (1.0 / F) * ((2.0 * F + 4.0 * F_Phi * F_Phi) / K)
end


Base.@kwdef struct InitialPhiSettings
    search_bounds::Tuple{Float64,Float64} = (-1.0, 1.0)
    residual_atol::Float64 = 1e-10
end


struct InitialPhiSolveError <: Exception
    status::Symbol
    bounds::Tuple{Float64,Float64}
    detail::String
end


function Base.showerror(io::IO, error::InitialPhiSolveError)
    print(io, "initial-field root solve failed (status=", error.status,
        ", bounds=", error.bounds, "): ", error.detail)
end


"""
Residual that fixes the scalar-field value at the beginning of the background.

This is the single model-specific hook for the early-time condition. The root
search and background pipeline remain unchanged if a different scalar
observable is imposed here in the future.
"""
@inline function initial_phi_residual(p::CosmoParams, Phi)
    F, _, _ = p.F_funcs(Phi)
    return F - 1.0
end


function _validate_initial_phi_settings(settings::InitialPhiSettings)
    lower, upper = settings.search_bounds
    isfinite(lower) && isfinite(upper) && lower < upper ||
        throw(ArgumentError("initial-field search bounds must be finite and ordered"))
    isfinite(settings.residual_atol) && settings.residual_atol > 0.0 ||
        throw(ArgumentError("initial-field residual tolerance must be finite and positive"))
    return settings
end


function _initial_phi_residual_value(p::CosmoParams, Phi, settings::InitialPhiSettings)
    local residual
    try
        residual = initial_phi_residual(p, Phi)
    catch error
        error isa InterruptException && rethrow()
        error isa InitialPhiSolveError && rethrow()
        throw(InitialPhiSolveError(
            :evaluation_failed,
            settings.search_bounds,
            "condition evaluation at Phi=$(Float64(Phi)) raised $(sprint(showerror, error))",
        ))
    end
    if !(residual isa Real && isfinite(residual))
        throw(InitialPhiSolveError(
            :nonfinite_residual,
            settings.search_bounds,
            "condition evaluation at Phi=$(Float64(Phi)) returned $residual",
        ))
    end
    return Float64(residual)
end


"""
Find the admissible early-time scalar value in a bounded interval.

All roots are searched so tangential roots are retained. The root closest to
zero is selected; equal-distance roots are ordered by their signed value, so
the negative root wins deterministically.
"""
function find_initial_phi(p::CosmoParams; settings=InitialPhiSettings())
    _validate_initial_phi_settings(settings)
    lower, upper = settings.search_bounds

    if lower <= 0.0 <= upper
        residual_at_zero = _initial_phi_residual_value(p, 0.0, settings)
        abs(residual_at_zero) <= settings.residual_atol && return 0.0
    end

    roots = try
        Roots.find_zeros(
            Phi -> _initial_phi_residual_value(p, Phi, settings),
            lower,
            upper,
        )
    catch error
        error isa InterruptException && rethrow()
        error isa InitialPhiSolveError && rethrow()
        throw(InitialPhiSolveError(
            :root_search_failed,
            settings.search_bounds,
            sprint(showerror, error),
        ))
    end

    valid_roots = Float64[]
    for root in roots
        root_value = Float64(root)
        isfinite(root_value) && lower <= root_value <= upper || continue
        residual = _initial_phi_residual_value(p, root_value, settings)
        abs(residual) <= settings.residual_atol || continue
        push!(valid_roots, root_value)
    end
    isempty(valid_roots) && throw(InitialPhiSolveError(
        :no_root,
        settings.search_bounds,
        "no root satisfies the residual tolerance $(settings.residual_atol)",
    ))

    sort!(valid_roots; by=Phi -> (abs(Phi), Phi))
    return first(valid_roots)
end


function setup_cosmo_cache(p_dummy::CosmoParams,
                           h_dummy::Float64,
                           dPhi_ini_dummy::Float64;
                           Phi_ini=0.0,
                           N_ini=-21.0,
                           shooting_reltol=1e-8,
                           shooting_abstol=1e-10,
                           full_reltol=1e-8,
                           full_abstol=1e-10,
                           maxiters=10_000)
    isfinite(h_dummy) && h_dummy > 0.0 ||
        throw(DomainError(h_dummy, "h_dummy must be finite and positive"))
    isfinite(dPhi_ini_dummy) ||
        throw(DomainError(dPhi_ini_dummy, "initial field velocity must be finite"))
    isfinite(Phi_ini) || throw(DomainError(Phi_ini, "initial field must be finite"))
    isfinite(N_ini) && N_ini < 0.0 ||
        throw(DomainError(N_ini, "N_ini must be finite and negative"))

    Or0_dummy = 1e-5
    shooting_integrator = setup_shooting_integrator(
        p_dummy,
        Or0_dummy,
        dPhi_ini_dummy;
        Phi_ini=Phi_ini,
        N_ini=N_ini,
        reltol=shooting_reltol,
        abstol=shooting_abstol,
        maxiters=maxiters,
    )

    full_initial_state = @SVector [Float64(Phi_ini), Float64(dPhi_ini_dummy), 0.0]
    full_context = EvolutionContext(
        p_dummy, Or0_dummy, PHYSICS_OK, PHYSICS_OK, false, 0)
    full_problem = ODEProblem(
        evolution_ode_full,
        full_initial_state,
        (Float64(N_ini), 0.0),
        full_context,
    )
    full_integrator = init(
        full_problem,
        Tsit5(),
        reltol=full_reltol,
        abstol=full_abstol,
        save_everystep=true,
        maxiters=maxiters,
        isoutofdomain=outside_physical_domain,
        unstable_check=(dt, u, p, t) -> !all(isfinite, u),
        verbose=SciMLLogging.None(),
    )
    sizehint!(full_integrator.sol.t, 3000)
    sizehint!(full_integrator.sol.u, 3000)
    if hasproperty(full_integrator.sol, :k) && full_integrator.sol.k !== nothing
        sizehint!(full_integrator.sol.k, 3000)
    end

    segbuf = alloc_segbuf(Float64, Float64, Float64)
    dummy_background = Background(
        full_integrator.sol,
        p_dummy,
        100.0 * h_dummy,
        0.0,
        0.0,
        0.0,
        Or0_dummy,
        nothing,
        0.0,
        segbuf,
    )
    growth_context = GrowthContext{typeof(dummy_background)}(nothing)
    N_growth_ini = -log1p(100.0)
    a_growth_ini = exp(N_growth_ini)
    growth_initial_state = @SVector [a_growth_ini, a_growth_ini]
    growth_problem = ODEProblem(
        growth_ode,
        growth_initial_state,
        (N_growth_ini, 0.0),
        growth_context,
    )
    growth_integrator = init(
        growth_problem,
        Tsit5(),
        reltol=1e-5,
        abstol=1e-7,
        save_everystep=false,
        maxiters=maxiters,
        unstable_check=(dt, u, p, t) -> !all(isfinite, u),
        verbose=SciMLLogging.None(),
    )

    result_ref = Ref{Union{Nothing,ShootingRootResult}}(nothing)
    return CosmoCache(
        shooting_integrator,
        full_integrator,
        growth_integrator,
        segbuf,
        result_ref,
    )
end


function radiation_density_parameter(h::Float64)
    omega_gamma = 2.47739207701433e-5
    Neff = 3.044
    neutrino_ratio = 7.0 / 8.0 * (4.0 / 11.0)^(4.0 / 3.0)
    omega_r = (1.0 + neutrino_ratio * Neff) * omega_gamma
    return omega_r / h^2
end


"""
Construct a background with a shared cache.

With the default `Phi_ini=nothing`, the initial field is obtained from
`find_initial_phi` before either the `U0` shooting solve or the full solve. A
finite numeric `Phi_ini` deliberately bypasses that search for diagnostics.
"""
function Background(cache::CosmoCache,
                    p_base::CosmoParams,
                    h::Float64,
                    Ob0::Float64,
                    dPhi_ini::Float64;
                    Phi_ini=nothing,
                    initial_phi_settings=InitialPhiSettings(),
                    shooting_settings=ShootingSettings())
    isfinite(h) && h > 0.0 ||
        throw(DomainError(h, "h must be finite and positive"))
    isfinite(Ob0) && Ob0 >= 0.0 ||
        throw(DomainError(Ob0, "Ob0 must be finite and non-negative"))
    isfinite(dPhi_ini) ||
        throw(DomainError(dPhi_ini, "initial field velocity must be finite"))
    resolved_Phi_ini = if isnothing(Phi_ini)
        find_initial_phi(p_base; settings=initial_phi_settings)
    else
        isfinite(Phi_ini) || throw(DomainError(Phi_ini, "initial field must be finite"))
        Float64(Phi_ini)
    end

    H0 = 100.0 * h
    omega_b = Ob0 * h^2
    omega_m = p_base.Om0 * h^2
    omega_gamma = 2.47739207701433e-5
    Or0 = radiation_density_parameter(h)

    shooting_result = find_U0!(
        cache.shooting_integrator,
        p_base,
        Or0;
        Phi_ini=resolved_Phi_ini,
        X_ini=dPhi_ini,
        settings=shooting_settings,
    )
    cache.last_shooting_result[] = shooting_result
    if !shooting_success(shooting_result)
        throw(BackgroundSolveError(
            BACKGROUND_SHOOTING_FAILURE,
            shooting_result.last_trial.physical_status,
            shooting_result.last_trial.retcode,
            "$(shooting_result.status) after $(shooting_result.trial_count) trials",
        ))
    end

    p_final = CosmoParams(
        p_base.Om0,
        shooting_result.U0,
        p_base.F_funcs,
        p_base.U_funcs,
    )
    full_context = cache.full_integrator.p
    full_context.cosmo = p_final
    full_context.Or0 = Or0
    full_context.last_physical_status = PHYSICS_OK
    full_context.stage_physical_status = PHYSICS_OK
    full_context.saw_invalid_stage = false
    full_context.domain_rejections = 0
    reinit!(
        cache.full_integrator,
        @SVector [resolved_Phi_ini, Float64(dPhi_ini), 0.0];
        erase_sol=true,
        reset_dt=true,
    )

    try
        solve!(cache.full_integrator)
    catch error
        error isa InterruptException && rethrow()
        if error isa PhysicalDomainError
            throw(BackgroundSolveError(
                BACKGROUND_FULL_PHYSICAL_FAILURE,
                error.status,
                cache.full_integrator.sol.retcode,
                sprint(showerror, error),
            ))
        end
        rethrow()
    end

    if !SciMLBase.successful_retcode(cache.full_integrator.sol)
        physical_status = full_context.last_physical_status
        status = physical_status == PHYSICS_OK ?
            BACKGROUND_FULL_ODE_FAILURE : BACKGROUND_FULL_PHYSICAL_FAILURE
        throw(BackgroundSolveError(
            status,
            physical_status,
            cache.full_integrator.sol.retcode,
            "full integration did not reach N=0",
        ))
    end

    solution = cache.full_integrator.sol
    state_today = solution(0.0)
    if !all(isfinite, state_today)
        throw(BackgroundSolveError(
            BACKGROUND_NONFINITE_SOLUTION,
            PHYSICS_NONFINITE_STATE,
            solution.retcode,
            "non-finite state at N=0",
        ))
    end

    present = inspect_friedmann(
        p_final, Or0, 0.0, state_today[1], state_today[2])
    if present.status != PHYSICS_OK
        throw(BackgroundSolveError(
            BACKGROUND_FULL_PHYSICAL_FAILURE,
            present.status,
            solution.retcode,
            "present-day state is outside the physical branch",
        ))
    end
    # q is known to be positive here, so log is safe and gives the same closure
    # root as q-1 with better scaling away from q=1.
    closure_residual = log(present.q)
    closure_tolerance = max(5.0 * shooting_settings.closure_atol, 1e-7)
    if !(isfinite(closure_residual) && abs(closure_residual) <= closure_tolerance)
        throw(BackgroundSolveError(
            BACKGROUND_CLOSURE_FAILURE,
            PHYSICS_OK,
            solution.retcode,
            "log(E(0)^2)=$closure_residual exceeds $closure_tolerance",
        ))
    end

    eta_today = state_today[3]
    if !isfinite(eta_today)
        throw(BackgroundSolveError(
            BACKGROUND_NONFINITE_SOLUTION,
            PHYSICS_NONFINITE_STATE,
            solution.retcode,
            "non-finite conformal time at N=0",
        ))
    end

    background_for_growth = Background(
        solution,
        p_final,
        H0,
        omega_b,
        omega_m,
        omega_gamma,
        Or0,
        nothing,
        eta_today,
        cache.segbuf,
    )
    growth_context = cache.growth_integrator.p
    growth_context.background = background_for_growth

    return Background(
        solution,
        p_final,
        H0,
        omega_b,
        omega_m,
        omega_gamma,
        Or0,
        cache.growth_integrator,
        eta_today,
        cache.segbuf,
    )
end


get_N(z) = -log1p(z)


function Phi_and_dPhi(bg::Background, z)
    state = bg.sol(get_N(z))
    return state[1], state[2]
end


Φ_and_dΦ(bg::Background, z) = Phi_and_dPhi(bg, z)


@inline function _E_from_fields(bg::Background, Phi, dPhi, N)
    result = friedmann_quantities(bg.p, bg.Or0, N, Phi, dPhi)
    return sqrt(result.q)
end


function E(bg::Background, z)
    Phi, dPhi = Phi_and_dPhi(bg, z)
    return _E_from_fields(bg, Phi, dPhi, get_N(z))
end


H(bg::Background, z) = bg.H0 * E(bg, z)


function mu_G(bg::Background, z)
    Phi, _ = Phi_and_dPhi(bg, z)
    F, F_Phi, _ = bg.p.F_funcs(Phi)
    return effective_gravity(F, F_Phi)
end


μ_G(bg::Background, z) = mu_G(bg, z)


function comoving_distance(bg::Background, z)
    N = get_N(z)
    eta_z = bg.sol(N)[3]
    return (c_kms / bg.H0) * (bg.eta_today - eta_z)
end


function observables(bg::Background, z)
    N = get_N(z)
    state = bg.sol(N)
    Phi = state[1]
    dPhi = state[2]
    eta_z = state[3]

    E_value = _E_from_fields(bg, Phi, dPhi, N)
    F, F_Phi, _ = bg.p.F_funcs(Phi)
    return (
        H=bg.H0 * E_value,
        μ_G=effective_gravity(F, F_Phi),
        comoving_distance=(c_kms / bg.H0) * (bg.eta_today - eta_z),
    )
end


function Gdot_H0_today(bg::Background)
    Phi0, dPhi0 = Phi_and_dPhi(bg, 0.0)
    mu_field = function (Phi)
        F, F_Phi, _ = bg.p.F_funcs(Phi)
        return effective_gravity(F, F_Phi)
    end
    dmu_dPhi = ForwardDiff.derivative(mu_field, Phi0)
    dmu_dz = -dmu_dPhi * dPhi0
    return -dmu_dz
end


# H0 = 100h km/s/Mpc in 1/yr (Julian year, 1 Mpc = 3.0856775814913673e19 km).
const H0_PER_YR_OVER_h = 100.0 * 365.25 * 86400.0 / 3.0856775814913673e19

"""
    Gdot_over_G_per_yr(bg) -> Float64

Ġ_eff/G_eff today in 1/yr, -H0 d ln μ_G/dz at z = 0. Valid for any normalization of μ_G.

Generated with Claude Opus 5.5.
"""
Gdot_over_G_per_yr(bg::Background) =
    H0_PER_YR_OVER_h * (bg.H0 / 100.0) * Gdot_H0_today(bg) / mu_G(bg, 0.0)


"""
    ppn_gamma_minus_one(bg) -> Float64

PPN γ - 1 = -F_Φ² / (F + 2F_Φ²) at the present-day field value.

Generated with Claude Opus 5.5.
"""
function ppn_gamma_minus_one(bg::Background)
    Phi0, _ = Phi_and_dPhi(bg, 0.0)
    F, F_Phi, _ = bg.p.F_funcs(Phi0)
    F_Phi2 = F_Phi * F_Phi
    return -F_Phi2 / (F + 2.0 * F_Phi2)
end


luminosity_distance(bg::Background, z_cmb, z_hel=z_cmb) =
    (1.0 + z_hel) * comoving_distance(bg, z_cmb)


function luminosity_distance_and_μG(bg::Background, z_cmb, z_hel)
    N = get_N(z_cmb)
    state = bg.sol(N)
    distance = (c_kms / bg.H0) * (bg.eta_today - state[3])
    F, F_Phi, _ = bg.p.F_funcs(state[1])
    return (1.0 + z_hel) * distance, effective_gravity(F, F_Phi)
end


function sound_horizon(bg::Background, z)
    R = 0.75 * bg.omega_b / bg.omega_gamma
    sound_speed_prefactor = c_kms / sqrt(3.0)
    integrand = function (a)
        zp = inv(a) - 1.0
        sound_speed = sound_speed_prefactor / sqrt(1.0 + R * a)
        return sound_speed / (H(bg, zp) * a * a)
    end
    a_max = inv(1.0 + z)
    value, _ = quadgk(integrand, 0.0, a_max; rtol=1e-5, segbuf=bg.segbuf)
    return value
end


function z_star(bg::Background)
    return (391.672 * bg.omega_m^(-0.372296) +
            937.422 * bg.omega_b^(-0.97966)) /
           (bg.omega_m^(-0.0192951) * bg.omega_b^(-0.93681)) +
           bg.omega_m^(-0.731631)
end


function z_drag(bg::Background)
    return (1.0 + 428.169 * bg.omega_b^(0.256459) * bg.omega_m^(0.616388) +
            925.56 * bg.omega_m^(0.751615)) / bg.omega_m^(0.714129)
end


function growth_ode(y, context::GrowthContext, N)
    isnothing(context.background) && return @SVector [0.0, 0.0]
    bg = context.background
    state = bg.sol(N)
    result = compute_field_dynamics(bg.p, bg.Or0, N, state[1], state[2])
    mu_value = effective_gravity(result.F, result.F_Phi)
    friction = 2.0 + result.s
    source = 1.5 * bg.p.Om0 * mu_value * result.a_inv3 * result.invq
    return @SVector [y[2], source * y[1] - friction * y[2]]
end


function compute_fσ8!(fs8_models::Vector{Float64},
                      bg::Background,
                      N_obs_sorted::Vector{Float64},
                      sort_perm::Vector{Int},
                      sigma8_0::Float64)
    length(fs8_models) == length(N_obs_sorted) == length(sort_perm) ||
        throw(DimensionMismatch("growth arrays must have equal length"))
    issorted(N_obs_sorted) ||
        throw(ArgumentError("N_obs_sorted must be sorted"))

    integrator = bg.growth_integrator
    reinit!(integrator; erase_sol=true, reset_dt=true)
    for i in eachindex(N_obs_sorted)
        target = N_obs_sorted[i]
        while integrator.t < target
            previous_t = integrator.t
            step!(integrator)
            SciMLBase.successful_retcode(integrator.sol) &&
                integrator.t > previous_t ||
                throw(ErrorException("growth integration failed before N=$target"))
        end
        fs8_models[sort_perm[i]] = integrator(target)[2]
    end

    while integrator.t < 0.0
        previous_t = integrator.t
        step!(integrator)
        SciMLBase.successful_retcode(integrator.sol) &&
            integrator.t > previous_t ||
            throw(ErrorException("growth integration failed before N=0"))
    end
    delta0 = integrator(0.0)[1]
    isfinite(delta0) && delta0 != 0.0 ||
        throw(ErrorException("invalid present-day growth normalization"))
    for i in eachindex(fs8_models)
        fs8_models[i] = (fs8_models[i] / delta0) * sigma8_0
    end
    return fs8_models
end
