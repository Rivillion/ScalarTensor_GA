"""
    WzBinning

Reader for the w(z) posteriors that `binning/python/run_nautilus.py` writes (DESI DR2 II Fig. 12 replica; binned
w(z) of Lodha et al. 2025, Eq. 13, and CPL). It holds the w(z) models, `load_wz_chain` and the summaries used by
`binning/WzPlots.jl`. Sampling, physics and likelihoods live in Python (`binning/python/wz/`).

By default a chain holds the weighted Nautilus points (n_eff ≈ 10⁴) rather than the smaller equal-weight resample,
and every summary (quantiles, means, the violins and bands of `WzPlots`) uses their weights.

It needs only HDF5 and Statistics from the root environment, so it loads in `plotting.ipynb` next to the GA stack.
Import it selectively (`using .WzBinning: load_wz_chain`), because the GA defines other names in `Main`.

Generated with Claude Opus 5.5.
"""
module WzBinning

using HDF5
using Statistics

export DarkEnergyModel, LCDMModel, CPLModel, BinnedWModel, w_param_names, w_of_z, model_tag
export WzChain, load_wz_chain, param_column, w_samples, w_quantiles, bestfit_w, binned_w_summary, param_summary
export weighted_quantile, weighted_mean, weighted_std, effective_sample_size, weighted_bandwidth

# ---------------------------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------------------------

"""
Abstract supertype of the w(z) models; each defines `w_param_names`, `w_of_z` and `model_tag`.

Generated with Claude Opus 5.5.
"""
abstract type DarkEnergyModel end

"""Cosmological constant, w = -1. Generated with Claude Opus 5.5."""
struct LCDMModel <: DarkEnergyModel end

"""CPL, w(z) = w0 + wa z/(1+z). Generated with Claude Opus 5.5."""
struct CPLModel <: DarkEnergyModel end

"""
    BinnedWModel(edges, smoothing)

w(z) = w_1 + Σ_i (w_{i+1} - w_i)/2 [1 + tanh((z - z_i)/s)] with transition redshifts `edges` (the last amplitude
holds above the last edge). DESI DR2 II Fig. 12 uses edges 0.7, 1.4, 2.1 and s = 0.02.

Generated with Claude Opus 5.5.
"""
struct BinnedWModel <: DarkEnergyModel
    edges::Vector{Float64}
    smoothing::Float64
end

w_param_names(::LCDMModel) = String[]
w_param_names(::CPLModel) = ["w0", "wa"]
w_param_names(model::BinnedWModel) = ["w[$i]" for i in 1:length(model.edges)+1]

"""
    w_of_z(model, θw, z)

Equation of state at redshift `z` for the parameters `θw`, ordered as `w_param_names(model)`.

Generated with Claude Opus 5.5.
"""
w_of_z(::LCDMModel, θw, z) = -one(float(z))
w_of_z(::CPLModel, θw, z) = θw[1] + θw[2] * z / (1 + z)

function w_of_z(model::BinnedWModel, θw, z)
    w = θw[1] + zero(z)
    for i in eachindex(model.edges)
        w += (θw[i+1] - θw[i]) / 2 * (1 + tanh((z - model.edges[i]) / model.smoothing))
    end
    return w
end

model_tag(::LCDMModel) = "lcdm"
model_tag(::CPLModel) = "cpl"
model_tag(::BinnedWModel) = "binned"

# ---------------------------------------------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------------------------------------------

"""
    WzChain

Posterior points of one model with their weights, as written by `run_nautilus.py`.

Fields: `model`, `param_names`, `samples` (points × parameters), `weights` (per point, normalised to sum 1),
`loglike` (per point), `weighted` (true for the weighted Nautilus points, false for the equal-weight resample),
`map_params`, `map_chi2`, `map_chi2_components` (χ² per likelihood at the MAP), `likelihoods`, `attributes` (seed,
n_live, n_eff, `lcdm_map_chi2`, timestamps, ...).

Generated with Claude Opus 5.5.
"""
struct WzChain{M<:DarkEnergyModel}
    model::M
    param_names::Vector{String}
    samples::Matrix{Float64}
    weights::Vector{Float64}
    loglike::Vector{Float64}
    weighted::Bool
    map_params::Dict{String,Float64}
    map_chi2::Float64
    map_chi2_components::Dict{String,Float64}
    likelihoods::Vector{Symbol}
    attributes::Dict{String,Any}
end

function Base.show(io::IO, chain::WzChain)
    kind = chain.weighted ? "weighted points" : "equal-weight draws"
    print(io, "WzChain(", model_tag(chain.model), ", ", size(chain.samples, 1), " ", kind, " (n_eff ",
        round(Int, effective_sample_size(chain.weights)), ") × ", length(chain.param_names), " parameters, χ²_MAP = ",
        round(chain.map_chi2; digits=3), ")")
end

"""
    load_wz_chain(path; weighted=true) -> WzChain

Read a chain file of `run_nautilus.py`. With `weighted=true` (the default) the chain holds the weighted Nautilus
points of the `nautilus/` group, with weights exp(log_w) normalised to sum 1. Files without that group, or
`weighted=false`, give the equal-weight resample (`samples`) with uniform weights.

Python writes row-major, so HDF5.jl reads `samples` (1, n_param, n_draw) as n_draw × n_param × 1 and
`nautilus/points` (n_point, n_param) as n_param × n_point.

Generated with Claude Opus 5.5.
"""
function load_wz_chain(path::AbstractString; weighted::Bool=true)
    h5open(path, "r") do file
        meta = Dict{String,Any}(k => read_attribute(file, k) for k in keys(attrs(file)))
        tag = pop!(meta, "model")
        model = tag == "binned" ? BinnedWModel(pop!(meta, "edges"), pop!(meta, "smoothing")) :
                tag == "cpl" ? CPLModel() : tag == "lcdm" ? LCDMModel() :
                throw(ArgumentError("unknown model tag $(repr(tag))"))
        use_weighted = weighted && haskey(file, "nautilus")
        if use_weighted
            samples = permutedims(read(file["nautilus/points"]))
            log_w = vec(read(file["nautilus/log_w"]))
            weights = exp.(log_w .- maximum(log_w))
            weights ./= sum(weights)
            loglike = vec(read(file["nautilus/log_l"]))
        else
            samples = read(file["samples"])[:, :, 1]
            weights = fill(1 / size(samples, 1), size(samples, 1))
            loglike = vec(read(file["logpost"]))
        end
        names = read(file["param_names"])
        size(samples, 2) == length(names) ||
            throw(DimensionMismatch("$(size(samples, 2)) sample columns for $(length(names)) parameters in $path"))
        map_params = Dict(zip(read(file["map/names"]), read(file["map/values"])))
        comps = Dict(zip(read(file["map/chi2_names"]), read(file["map/chi2_values"])))
        return WzChain(model, names, samples, weights, loglike, use_weighted, map_params, sum(values(comps)),
            comps, Symbol.(read(file["likelihoods"])), meta)
    end
end

# ---------------------------------------------------------------------------------------------------------------
# Weighted statistics
# ---------------------------------------------------------------------------------------------------------------

"""
    weighted_quantile(x, w, probs) -> Vector

Quantiles of `x` with weights `w` (any non-negative scale). Each point stands at the midpoint of its weight in the
cumulative distribution, F_k = (Σ_{j≤k} w_j - w_k/2) / Σ w, and the quantile is linearly interpolated in F. For
uniform weights this is the Hazen definition, (k - 1/2)/n.

Generated with Claude Opus 5.5.
"""
function weighted_quantile(x::AbstractVector{<:Real}, w::AbstractVector{<:Real}, probs)
    length(x) == length(w) || throw(DimensionMismatch("$(length(x)) values but $(length(w)) weights"))
    isempty(x) && throw(ArgumentError("weighted_quantile of an empty vector"))
    order = sortperm(x)
    xs = x[order]
    ws = float.(w[order])
    cdf = cumsum(ws)
    total = cdf[end]
    total > 0 || throw(ArgumentError("weights sum to $total"))
    cdf .= (cdf .- ws ./ 2) ./ total
    return map(probs) do p
        0 <= p <= 1 || throw(ArgumentError("probability $p outside [0, 1]"))
        k = searchsortedfirst(cdf, p)
        k == 1 && return float(xs[1])
        k > length(cdf) && return float(xs[end])
        t = (p - cdf[k-1]) / (cdf[k] - cdf[k-1])
        return xs[k-1] + t * (xs[k] - xs[k-1])
    end
end

"""
    weighted_mean(x, w)

Mean of `x` with weights `w`.

Generated with Claude Opus 5.5.
"""
weighted_mean(x::AbstractVector{<:Real}, w::AbstractVector{<:Real}) = sum(w .* x) / sum(w)

"""
    weighted_std(x, w)

Standard deviation of `x` with weights `w`, corrected for the effective sample size: Σw(x - μ)² / (Σw - Σw²/Σw),
which reduces to the usual n - 1 estimator for uniform weights.

Generated with Claude Opus 5.5.
"""
function weighted_std(x::AbstractVector{<:Real}, w::AbstractVector{<:Real})
    μ = weighted_mean(x, w)
    total = sum(w)
    return sqrt(sum(w .* (x .- μ) .^ 2) / (total - sum(abs2, w) / total))
end

"""
    effective_sample_size(w)

Kish effective sample size (Σw)² / Σw².

Generated with Claude Opus 5.5.
"""
effective_sample_size(w::AbstractVector{<:Real}) = sum(w)^2 / sum(abs2, w)

"""
    weighted_bandwidth(x, w; alpha=0.9)

Silverman's rule of thumb, alpha · min(σ, IQR/1.34) · n_eff^(-1/5), with the weighted σ and IQR and the Kish n_eff.
KernelDensity's default bandwidth ignores the weights, so near-zero-weight points far in the prior tails would
inflate it.

Generated with Claude Opus 5.5.
"""
function weighted_bandwidth(x::AbstractVector{<:Real}, w::AbstractVector{<:Real}; alpha=0.9)
    q25, q75 = weighted_quantile(x, w, (0.25, 0.75))
    width = min(weighted_std(x, w), (q75 - q25) / 1.34)
    return alpha * width * effective_sample_size(w)^(-0.2)
end

# ---------------------------------------------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------------------------------------------

"""
    param_column(chain, name) -> Vector{Float64}

All points of the parameter `name`, aligned with `chain.weights`.

Generated with Claude Opus 5.5.
"""
function param_column(chain::WzChain, name::AbstractString)
    index = findfirst(==(name), chain.param_names)
    index === nothing && throw(ArgumentError("parameter $name not in chain ($(join(chain.param_names, ", ")))"))
    return chain.samples[:, index]
end

"""
    w_samples(chain, z) -> Matrix

w(z) of every point (rows, aligned with `chain.weights`) on the redshift grid `z` (columns).

Generated with Claude Opus 5.5.
"""
function w_samples(chain::WzChain, z::AbstractVector)
    columns = [findfirst(==(name), chain.param_names) for name in w_param_names(chain.model)]
    any(isnothing, columns) && throw(ArgumentError("chain lacks some of $(w_param_names(chain.model))"))
    θ = chain.samples[:, columns]
    return [w_of_z(chain.model, view(θ, row, :), zj) for row in axes(θ, 1), zj in z]
end

"""
    w_quantiles(chain, z; probs=[0.025, 0.16, 0.5, 0.84, 0.975]) -> Matrix

Point-by-point weighted posterior quantiles of w(z): one row per probability, one column per redshift. The default
rows are the 95% band, the 68% band and the median.

Generated with Claude Opus 5.5.
"""
function w_quantiles(chain::WzChain, z::AbstractVector; probs=[0.025, 0.16, 0.5, 0.84, 0.975])
    W = w_samples(chain, z)
    return reduce(hcat, [weighted_quantile(view(W, :, j), chain.weights, probs) for j in axes(W, 2)])
end

"""
    bestfit_w(chain, z) -> Vector

w(z) at the MAP point.

Generated with Claude Opus 5.5.
"""
function bestfit_w(chain::WzChain, z::AbstractVector)
    θw = [chain.map_params[name] for name in w_param_names(chain.model)]
    return [w_of_z(chain.model, θw, zi) for zi in z]
end

"""
    binned_w_summary(chain; probs=(0.16, 0.5, 0.84)) -> Vector{NamedTuple}

One entry per bin of a binned chain: `index`, `z_lo`, `z_hi` (`Inf` for the last bin), `lo`, `median`, `hi` (the
quantiles in `probs`), `mean`, `std`, `map`, and the bin's `samples` with their `weights`. All statistics are
weighted.

Generated with Claude Opus 5.5.
"""
function binned_w_summary(chain::WzChain; probs=(0.16, 0.5, 0.84))
    chain.model isa BinnedWModel || throw(ArgumentError("binned_w_summary needs a binned chain"))
    edges = vcat(0.0, chain.model.edges, Inf)
    w = chain.weights
    return map(enumerate(w_param_names(chain.model))) do (i, name)
        samples = param_column(chain, name)
        lo, med, hi = weighted_quantile(samples, w, collect(probs))
        (index=i, z_lo=edges[i], z_hi=edges[i+1], lo=lo, median=med, hi=hi, mean=weighted_mean(samples, w),
            std=weighted_std(samples, w), map=chain.map_params[name], samples=samples, weights=w)
    end
end

"""
    param_summary(chain) -> Vector{NamedTuple}

Weighted mean, standard deviation, median, 68% and 95% intervals and MAP value of every parameter.

Generated with Claude Opus 5.5.
"""
function param_summary(chain::WzChain)
    w = chain.weights
    return map(enumerate(chain.param_names)) do (k, name)
        x = chain.samples[:, k]
        q = weighted_quantile(x, w, [0.025, 0.16, 0.5, 0.84, 0.975])
        (name=name, mean=weighted_mean(x, w), std=weighted_std(x, w), median=q[3], lo68=q[2], hi68=q[4],
            lo95=q[1], hi95=q[5], map=get(chain.map_params, name, NaN))
    end
end

end # module
