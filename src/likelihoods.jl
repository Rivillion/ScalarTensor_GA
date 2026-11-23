using SciMLBase
using LinearAlgebra

const c_kms = 299792.458

abstract type AbstractLikelihood end


# ===============================================
# Plain-text data readers (no CSV/DataFrames, which cost ~80 MB of resident memory per process)
# ===============================================
"""
    _data_rows(path; comment="#") -> Vector{Vector{SubString{String}}}

Whitespace-split fields of every non-blank line of `path`, skipping lines that start with `comment`
(`nothing` keeps them).

Generated with Claude Opus 5.5.
"""
function _data_rows(path::AbstractString; comment::Union{Nothing,String}="#")
    rows = Vector{Vector{SubString{String}}}()
    for line in eachline(path)
        stripped = strip(line)
        isempty(stripped) && continue
        comment !== nothing && startswith(stripped, comment) && continue
        push!(rows, split(stripped))
    end
    return rows
end

"""
    _read_table(path; header=nothing, comment="#") -> (columns::Vector{String}, rows)

A whitespace-separated table. Without `header`, the first data line is the header.

Generated with Claude Opus 5.5.
"""
function _read_table(path::AbstractString; header=nothing, comment::Union{Nothing,String}="#")
    rows = _data_rows(path; comment)
    if header === nothing
        isempty(rows) && throw(ArgumentError("$path has no header line"))
        return String.(rows[1]), rows[2:end]
    end
    return collect(String, header), rows
end

"""
    _column(columns, rows, name, T=Float64) -> Vector{T}

Parse the column `name` of a table read by `_read_table`. Rows shorter than the header are an error only when they
lack this column.

Generated with Claude Opus 5.5.
"""
function _column(columns::Vector{String}, rows, name::AbstractString, ::Type{T}=Float64) where {T}
    j = findfirst(==(name), columns)
    j === nothing && throw(ArgumentError("column $name not found; columns are $(join(columns, ", "))"))
    return T === String ? [String(row[j]) for row in rows] : [parse(T, row[j]) for row in rows]
end

"""
    _read_matrix(path; comment="#") -> Matrix{Float64}

A dense matrix written one row per line.

Generated with Claude Opus 5.5.
"""
function _read_matrix(path::AbstractString; comment::Union{Nothing,String}="#")
    rows = _data_rows(path; comment)
    n, m = length(rows), length(first(rows))
    all(row -> length(row) == m, rows) || throw(ArgumentError("$path has rows of unequal length"))
    matrix = Matrix{Float64}(undef, n, m)
    for i in 1:n, j in 1:m
        matrix[i, j] = parse(Float64, rows[i][j])
    end
    return matrix
end

"""
    _read_flat_covariance(path) -> Matrix{Float64}

Pantheon+ covariance format: the dimension `n` on the first line, then the `n²` entries one per line, read in
column-major order as `reshape(values, n, n)`. Streams the file instead of building a table.

Generated with Claude Opus 5.5.
"""
function _read_flat_covariance(path::AbstractString)
    open(path) do io
        n = parse(Int, strip(readline(io)))
        values = Vector{Float64}(undef, n * n)
        k = 0
        for line in eachline(io)
            stripped = strip(line)
            isempty(stripped) && continue
            k += 1
            k <= n * n || throw(ArgumentError("$path has more than $n² covariance entries"))
            values[k] = parse(Float64, stripped)
        end
        k == n * n || throw(ArgumentError("$path has $k covariance entries, expected $(n * n)"))
        return reshape(values, n, n)
    end
end

"""
    _read_pantheon_table(path) -> NamedTuple

The Pantheon+ data columns used by the likelihoods (header line, no comments).

Generated with Claude Opus 5.5.
"""
function _read_pantheon_table(path::AbstractString)
    columns, rows = _read_table(path; comment=nothing)
    return (
        zHD=_column(columns, rows, "zHD"),
        zHEL=_column(columns, rows, "zHEL"),
        m_b_corr=_column(columns, rows, "m_b_corr"),
        IS_CALIBRATOR=_column(columns, rows, "IS_CALIBRATOR", Int),
        CEPH_DIST=_column(columns, rows, "CEPH_DIST"),
    )
end


# ===============================================
# PantheonPlus + SH0ES
# ===============================================

struct PantheonPlusSH0ES <: AbstractLikelihood
    z_cmb::Vector{Float64}
    z_hel::Vector{Float64}
    N_cmb::Vector{Float64}
    zhel_plus_one::Vector{Float64}
    mb_obs::Vector{Float64}
    is_calibrator::Vector{Bool}
    mu_shoes::Vector{Float64}
    L_inv::LowerTriangular{Float64,Matrix{Float64}}
    diff_buffer::Vector{Float64}
    mul_buffer::Vector{Float64}
end

function PantheonPlusSH0ES(data_path::String, cov_path::String)
    df = _read_pantheon_table(data_path)

    mask = df.IS_CALIBRATOR .== 1 .|| df.zHD .> 0.0
    z_cmb = df.zHD[mask]
    z_hel = df.zHEL[mask]
    N_cmb = get_N.(z_cmb)
    zhel_plus_one = 1.0 .+ z_hel
    mb_obs = df.m_b_corr[mask]

    # Distinguish Cepheids from supernovae
    is_calibrator = Vector{Bool}(df.IS_CALIBRATOR[mask] .== 1)
    mu_shoes = df.CEPH_DIST[mask]

    # Load covariante matrix
    cov_full = _read_flat_covariance(cov_path)

    mask_idx = findall(mask)
    cov_masked = cov_full[mask_idx, mask_idx]

    # Cholesky decomposition
    L_fact = try
        cholesky(Symmetric(cov_masked))
    catch
        cholesky(Symmetric(cov_masked + I * 1e-8))
    end

    # Precompute the inverse of the lower triangular matrix
    L_inv = inv(LowerTriangular(L_fact.L))

    n_final = length(z_cmb)
    println("PantheonPlus+SH0ES initialized. Total SN: $n_final. Cepheid anchors: $(sum(is_calibrator))")

    # Pass the two preallocated zero arrays
    return PantheonPlusSH0ES(z_cmb, z_hel, N_cmb, zhel_plus_one, mb_obs, is_calibrator, mu_shoes,
        L_inv, zeros(n_final), zeros(n_final))
end

# Note: all functions are called log_likelihood.
# The only difference is that the `like` argument for each is different.
# Depending on the struct provided to log_likelihood, Julia can handle all types of structs automatically.
function log_likelihood(like::PantheonPlusSH0ES, bg::Background, M_nuisance::Float64,
    use_mu_G_correction=false, gamma=1.46)

    return use_mu_G_correction ?
           _log_likelihood_muG_correction(like, bg, M_nuisance, gamma) :
           _log_likelihood_no_muG_correction(like, bg, M_nuisance)
end


function _log_likelihood_muG_correction(like::PantheonPlusSH0ES, bg::Background,
    M_nuisance::Float64, gamma::Float64)

    distance_prefactor = c_kms / bg.H0
    correction_prefactor = 2.5 * gamma

    @inbounds for i in eachindex(like.z_cmb)
        state = bg.sol(like.N_cmb[i])
        Φ = state[1]
        η_z = state[3]

        dist = distance_prefactor * (bg.η_today - η_z)
        dL = like.zhel_plus_one[i] * dist

        F, F_Φ, _ = bg.p.F_funcs(Φ)
        F_Φ2 = F_Φ * F_Φ
        mu_g_val = (1.0 / F) * ((2.0 * F + 4.0 * F_Φ2) / (2.0 * F + 3.0 * F_Φ2))

        mu_cosmo = 5.0 * log10(dL) + 25.0

        mu_model = like.is_calibrator[i] ? like.mu_shoes[i] : mu_cosmo

        if mu_g_val <= 0.0 || isnan(mu_g_val)
            return -1e6
        end
        correction = correction_prefactor * log10(mu_g_val)

        like.diff_buffer[i] = like.mb_obs[i] - (mu_model + M_nuisance - correction)
    end

    # In-place matrix-vector multiplication: L_inv * diff_buffer -> mul_buffer
    mul!(like.mul_buffer, like.L_inv, like.diff_buffer)

    return -0.5 * dot(like.mul_buffer, like.mul_buffer)
end


function _log_likelihood_no_muG_correction(like::PantheonPlusSH0ES, bg::Background,
    M_nuisance::Float64)

    distance_prefactor = c_kms / bg.H0

    @inbounds for i in eachindex(like.N_cmb)
        state = bg.sol(like.N_cmb[i])
        η_z = state[3]

        dist = distance_prefactor * (bg.η_today - η_z)
        dL = like.zhel_plus_one[i] * dist
        mu_cosmo = 5.0 * log10(dL) + 25.0

        mu_model = like.is_calibrator[i] ? like.mu_shoes[i] : mu_cosmo
        like.diff_buffer[i] = like.mb_obs[i] - (mu_model + M_nuisance)
    end

    mul!(like.mul_buffer, like.L_inv, like.diff_buffer)
    return -0.5 * dot(like.mul_buffer, like.mul_buffer)
end


# ===============================================
# PantheonPlus + SH0ES with the mu_G luminosity correction
# ===============================================
"""
    PantheonPlusSH0ES_muG(base; gamma=1.46)

Pantheon+SH0ES with the supernova luminosity correction `2.5 γ log10 μ_G(z)` applied to every
entry, calibrators included. A separate type, so the stored likelihood names (`nameof(typeof(like))`)
tell corrected runs from uncorrected ones. `base` holds the data and the scratch buffers.

Generated with Claude Opus 5.5.
"""
struct PantheonPlusSH0ES_muG <: AbstractLikelihood
    base::PantheonPlusSH0ES
    gamma::Float64
end

PantheonPlusSH0ES_muG(base::PantheonPlusSH0ES; gamma=1.46) = PantheonPlusSH0ES_muG(base, gamma)

log_likelihood(like::PantheonPlusSH0ES_muG, bg::Background, M_nuisance::Float64) =
    _log_likelihood_muG_correction(like.base, bg, M_nuisance, like.gamma)

# Likelihoods that take the SN absolute magnitude M_nuisance as an extra argument.
const SH0ESLikelihood = Union{PantheonPlusSH0ES,PantheonPlusSH0ES_muG}


# ===============================================
# DESI DR 2 BAO
# ===============================================
struct DESI_BAO <: AbstractLikelihood
    z_obs::Vector{Float64}
    data_vector::Vector{Float64}
    flags::Vector{Int} # 1: DV_over_rd; 2: DM_over_DH; 3: DM_over_rd; 4: DH_over_rd
    L::LowerTriangular{Float64,Matrix{Float64}}
    diff_buffer::Vector{Float64}
end

function DESI_BAO(mean_path::String, cov_path::String)
    columns, rows = _read_table(mean_path; header=["z", "value", "quantity"])
    df = (z=_column(columns, rows, "z"), value=_column(columns, rows, "value"),
        quantity=_column(columns, rows, "quantity", String))

    # To select what data we are exporting.
    # DESI DR 2 reports either (DV_over_rd, DM_over_DH), or (DV_over_rd, DM_over_rd, DH_over_rd)
    flags = map(df.quantity) do q
        q == "DV_over_rs" ? 1 :
        q == "DM_over_DH" ? 2 :
        q == "DM_over_rs" ? 3 : 4
    end

    # Load covariance and prepare Cholesky
    cov = _read_matrix(cov_path)
    L_fact = try
        cholesky(Symmetric(cov))
    catch
        cholesky(Symmetric(cov + I * 1e-8))
    end

    n_final = length(df.z)
    println("DESI DR2 BAO initialized. $n_final data points.")
    return DESI_BAO(df.z, df.value, flags, LowerTriangular(L_fact.L), zeros(n_final))
end

function log_likelihood(like::DESI_BAO, bg::Background)
    zd = z_drag(bg)
    rs_drag = sound_horizon(bg, zd,)

    @inbounds for i in eachindex(like.z_obs)
        z = like.z_obs[i]
        flag = like.flags[i]

        obs = observables(bg, z)
        dm = obs.comoving_distance
        dh = c_kms / obs.H

        if flag == 1
            theory = cbrt(z * dm^2 * dh) / rs_drag
        elseif flag == 2
            theory = dm / dh
        elseif flag == 3
            theory = dm / rs_drag
        else
            theory = dh / rs_drag
        end

        like.diff_buffer[i] = like.data_vector[i] - theory
    end

    ldiv!(like.L, like.diff_buffer)
    return -0.5 * dot(like.diff_buffer, like.diff_buffer)
end


# ===============================================
# Redshift Space Distortion (fσ₈ data)
# ===============================================
struct GrowthRateLikelihood <: AbstractLikelihood
    z_obs::Vector{Float64}
    fs8_obs::Vector{Float64}
    fs8_err::Vector{Float64}
    idx_diag::Vector{Int}
    idx_wig::Vector{Int}
    inv_cov_wig::Matrix{Float64}
    diff_buffer::Vector{Float64}
    wig_buffer::Vector{Float64}      # Preallocated buffer for WiggleZ subset
    mul_buffer::Vector{Float64}      # Preallocated buffer for matrix multiplication
    fs8_models::Vector{Float64}      # Preallocated buffer for ODE predictions
    N_obs_sorted::Vector{Float64}    # Pre-sorted target e-folds
    sort_perm::Vector{Int}           # Index map to restore original order of data
    EDc_fid::Vector{Float64}         # Fiducial E(z) D_C(z), the denominator of the ratio (read-only)
    idx_boss::Vector{Int}            # The two correlated BOSS DR12 LRG points
    inv_cov_boss::Matrix{Float64}    # Their inverse 2×2 covariance (read-only)
end

"""
    fiducial_E_Dc(z, Om_fid) -> Float64

Dimensionless `E(z) D_C(z)` of a spatially flat LCDM model with matter density `Om_fid`, where
`D_C(z) = ∫₀ᶻ dz'/E(z')`. This equals `H(z) d_A(z) (1+z) / c`, which does not depend on `H0`, so `Om_fid` is the only
fiducial parameter needed. It is the denominator of the fiducial-cosmology correction of the growth data.

Generated with Claude Opus 5.5.
"""
function fiducial_E_Dc(z::Float64, Om_fid::Float64)
    E_fid(x) = sqrt(Om_fid * (1.0 + x)^3 + 1.0 - Om_fid)
    D_fid, _ = quadgk(x -> 1.0 / E_fid(x), 0.0, z; rtol=1e-10)
    return E_fid(z) * D_fid
end

"""
    boss_fs8_inverse_covariance(cov_path, z) -> Matrix{Float64}

Inverse 2×2 covariance of the BOSS DR12 LRG fσ8 points at z = 0.38 and 0.51: the fσ8 block of the SDSS 6×6 covariance
of (D_M/r_d, D_H/r_d, fσ8) at those redshifts, whose fσ8 entries are the 3rd and 6th.

Generated with Claude Opus 5.5.
"""
function boss_fs8_inverse_covariance(cov_path::String, z::Vector{Float64})
    z == [0.38, 0.51] || throw(ArgumentError("expected the BOSS block at z = 0.38 and 0.51, in that order; got $z"))
    cov_sdss = _read_matrix(cov_path)
    size(cov_sdss) == (6, 6) || throw(ArgumentError("$cov_path is not a 6×6 matrix"))
    return inv(cov_sdss[[3, 6], [3, 6]])
end

function GrowthRateLikelihood(data_path::String, cov_path::String, boss_cov_path::String)
    # The optional sixth column (source label) is not used.
    columns, rows = _read_table(data_path; header=["z", "fs8", "err", "corr_block", "Omega_m", "source"])
    df = (z=_column(columns, rows, "z"), fs8=_column(columns, rows, "fs8"),
        err=_column(columns, rows, "err"), corr_block=_column(columns, rows, "corr_block", Int),
        Om_fid=_column(columns, rows, "Omega_m"))
    EDc_fid = fiducial_E_Dc.(df.z, df.Om_fid)

    idx_diag = findall(==(0), df.corr_block)
    idx_wig = findall(==(1), df.corr_block)
    idx_boss = findall(==(2), df.corr_block)

    cov_wig = _read_matrix(cov_path)
    inv_cov_wig = inv(cov_wig)
    inv_cov_boss = boss_fs8_inverse_covariance(boss_cov_path, df.z[idx_boss])

    n_final = length(df.z)

    # Preallocate and sort e-folds for a single ODE evaluation
    N_obs = -log.(1.0 .+ df.z)
    sort_perm = sortperm(N_obs)
    N_obs_sorted = N_obs[sort_perm]

    println("Redshift Space Distortion initialized. $n_final points.")
    return GrowthRateLikelihood(df.z, df.fs8, df.err, idx_diag, idx_wig, inv_cov_wig,
        zeros(n_final), zeros(length(idx_wig)), zeros(length(idx_wig)),
        zeros(n_final), N_obs_sorted, sort_perm, EDc_fid, idx_boss, inv_cov_boss)
end

function log_likelihood(like::GrowthRateLikelihood, bg::Background, sigma8_0::Float64)
    # Compute all theoretical values once and the residuals

    compute_fσ8!(like.fs8_models, bg, like.N_obs_sorted, like.sort_perm, sigma8_0)
    @inbounds for i in eachindex(like.z_obs)
        # Fiducial-cosmology correction: ratiogrowth(z) = H d_A / (H_fid d_A_fid), dividing the prediction.
        # The (1+z) of d_A cancels in the ratio.
        obs = observables(bg, like.z_obs[i])
        ratiogrowth = obs.H * obs.comoving_distance / (c_kms * like.EDc_fid[i])
        like.diff_buffer[i] = like.fs8_obs[i] - like.fs8_models[i] / ratiogrowth
    end

    chi2 = 0.0
    # Uncorrelated data
    @inbounds for i in like.idx_diag
        chi2 += (like.diff_buffer[i] / like.fs8_err[i])^2
    end

    # Correlated WiggleZ data
    @inbounds for (k, i) in enumerate(like.idx_wig)
        like.wig_buffer[k] = like.diff_buffer[i]
    end
    mul!(like.mul_buffer, like.inv_cov_wig, like.wig_buffer)
    chi2 += dot(like.wig_buffer, like.mul_buffer)

    # Correlated BOSS DR12 LRG data
    @inbounds for (j, a) in enumerate(like.idx_boss), (k, b) in enumerate(like.idx_boss)
        chi2 += like.diff_buffer[a] * like.inv_cov_boss[j, k] * like.diff_buffer[b]
    end

    return -0.5 * chi2
end


# ===============================================
# Compressed CMB (Zhai et. al.)
# ===============================================
struct PlanckCompressed_Zhai2018 <: AbstractLikelihood
    obs_data::SVector{3,Float64}
    inv_cov::SMatrix{3,3,Float64,9}
end

function PlanckCompressed_Zhai2018()
    obs_data = @SVector[1.74963, 301.80845, 0.02237]
    cov_raw = 1.0e-8 .* [
        1598.9554 17112.007 -36.311179;
        17112.007 811208.45 -494.79813;
        -36.311179 -494.79813 2.1242182
    ]
    inv_cov = inv(SMatrix{3,3}(cov_raw))
    println("Planck 2018 (Zhai et. al.) initialized")
    return PlanckCompressed_Zhai2018(obs_data, inv_cov)
end

function log_likelihood(like::PlanckCompressed_Zhai2018, bg::Background)
    zs = z_star(bg)
    dm = comoving_distance(bg, zs)
    rs = sound_horizon(bg, zs)

    R = sqrt(bg.p.Om0 * bg.H0^2) * dm / c_kms
    la = π * dm / rs
    obh2 = bg.ωb

    diff = @SVector[R, la, obh2] - like.obs_data
    return -0.5 * dot(diff, like.inv_cov * diff)
end

# ===============================================
# Time variation of G
# ===============================================
struct GdotLikelihood <: AbstractLikelihood
    mean::Float64
    sigma::Float64
end

function GdotLikelihood(; mean=0.0, sigma=0.001)
    println("Gdot initialized: $mean +/- $sigma (dimensionless)")
    return GdotLikelihood(mean, sigma)
end

function log_likelihood(like::GdotLikelihood, bg::Background)
    Gdot = try
        Gdot_H0_today(bg)
    catch
        return -1e6
    end
    return -0.5 * ((Gdot - like.mean) / like.sigma)^2
end


# ===============================================
# Local constraint on G
# ===============================================
struct MuGLikelihood <: AbstractLikelihood
    mean::Float64
    sigma::Float64
end

function MuGLikelihood(; mean=1.0, sigma=1.2e-5)
    println("mu_G initialized: $mean +/- $sigma (dimensionless)")
    return MuGLikelihood(mean, sigma)
end

function log_likelihood(like::MuGLikelihood, bg::Background)
    mu_g = try
        μ_G(bg, 0.0)
    catch
        return -1e6
    end
    return -0.5 * ((mu_g - like.mean) / like.sigma)^2
end


# ===============================================
# Local tests of gravity, evaluated at the present-day field value
# ===============================================
"""
    PPNGammaLikelihood(; mean=2.1e-5, sigma=2.3e-5)

Cassini Shapiro-delay bound on the PPN parameter, γ - 1 = (2.1 ± 2.3) × 10⁻⁵ (Bertotti et al. 2003),
with γ - 1 = -F_Φ² / (F + 2F_Φ²) at Φ(z=0).

Generated with Claude Opus 5.5.
"""
struct PPNGammaLikelihood <: AbstractLikelihood
    mean::Float64
    sigma::Float64
end

function PPNGammaLikelihood(; mean=2.1e-5, sigma=2.3e-5)
    println("PPN gamma-1 initialized: $mean +/- $sigma (Cassini)")
    return PPNGammaLikelihood(mean, sigma)
end

log_likelihood(like::PPNGammaLikelihood, bg::Background) =
    -0.5 * ((ppn_gamma_minus_one(bg) - like.mean) / like.sigma)^2


"""
    LLRGdotLikelihood(; mean=-5.0e-15, sigma=9.6e-15)

Lunar Laser Ranging bound Ġ/G = (-5.0 ± 9.6) × 10⁻¹⁵ yr⁻¹ (Biskupek, Müller & Torre 2021), with
Ġ/G = -H0 d ln μ_G/dz at z = 0 in yr⁻¹.

Generated with Claude Opus 5.5.
"""
struct LLRGdotLikelihood <: AbstractLikelihood
    mean::Float64
    sigma::Float64
end

function LLRGdotLikelihood(; mean=-5.0e-15, sigma=9.6e-15)
    println("LLR Gdot/G initialized: $mean +/- $sigma (1/yr)")
    return LLRGdotLikelihood(mean, sigma)
end

log_likelihood(like::LLRGdotLikelihood, bg::Background) =
    -0.5 * ((Gdot_over_G_per_yr(bg) - like.mean) / like.sigma)^2


"""
    BBNGLikelihood(; mean=0.99, sigma=0.03)

BBN bound on G_BBN/G_0 = 1/μ_G(0) (Alvey et al. 2020, 0.99 +0.06/-0.05 at 2σ, symmetrized to σ = 0.03).
F(Φ_ini) = 1, so G during BBN is G_N.

Generated with Claude Opus 5.5.
"""
struct BBNGLikelihood <: AbstractLikelihood
    mean::Float64
    sigma::Float64
end

function BBNGLikelihood(; mean=0.99, sigma=0.03)
    println("BBN G_BBN/G_0 initialized: $mean +/- $sigma (dimensionless)")
    return BBNGLikelihood(mean, sigma)
end

log_likelihood(like::BBNGLikelihood, bg::Background) =
    -0.5 * ((1.0 / μ_G(bg, 0.0) - like.mean) / like.sigma)^2


# ===============================================
# Per-thread copies
# ===============================================
"""
    thread_copy(like::AbstractLikelihood) -> AbstractLikelihood

A copy of `like` for one worker thread: fresh scratch buffers, but the read-only data (redshifts, observations and
the dense covariance factors) are shared with `like` instead of copied. The likelihoods write only to their
buffers, so sharing the rest across threads is safe; the 1701×1701 Pantheon+ factor is then held once (22 MB)
instead of once per thread. Types without a specialised method fall back to `deepcopy`, which is cheap for them.

Generated with Claude Opus 5.5.
"""
thread_copy(like::AbstractLikelihood) = deepcopy(like)

thread_copy(like::PantheonPlusSH0ES) = PantheonPlusSH0ES(like.z_cmb, like.z_hel, like.N_cmb,
    like.zhel_plus_one, like.mb_obs, like.is_calibrator, like.mu_shoes, like.L_inv,
    similar(like.diff_buffer), similar(like.mul_buffer))

thread_copy(like::PantheonPlusSH0ES_muG) = PantheonPlusSH0ES_muG(thread_copy(like.base), like.gamma)
