"""
    WzPlots

Makie helpers that draw the Nautilus chains (read with `WzBinning.load_wz_chain`) on any `Axis`, so the same calls work in the Fig. 12 replica
(`binning/plot_fig12.jl`) and on the w_ST(z) axes of `plotting.ipynb`. Include `binning/WzBinning.jl` first:

    include("binning/WzBinning.jl"); include("binning/WzPlots.jl")
    using .WzBinning: load_wz_chain
    using .WzPlots: plot_w_cpl!, plot_w_binned!, figure_wz_fig12

Sizes and line widths default to the active Makie theme, so the notebook theme applies unchanged.

Generated with Claude Opus 5.5.
"""
module WzPlots

using CairoMakie
using LaTeXStrings
using ..WzBinning

export plot_w_cpl!, plot_w_binned!, figure_wz_fig12, errorbar_legend_handle

"""
    plot_w_cpl!(ax, chain; z=range(0, 4; length=400), color="#009E73", band_alpha=(0.15, 0.35),
                bestfit=true, label=L"w_0w_a\\mathrm{CDM}", linewidth=nothing)

Draw the 95% and 68% point-by-point bands of w(z) from a CPL chain (`band_alpha` gives their opacities) and the
best-fit (MAP) curve, as in DESI DR2 Fig. 12. With `bestfit=false` the curve is the posterior median instead. The
bands and the median are weighted quantiles (`w_quantiles`).
Works for any WzBinning chain, so it can also show the band of a binned chain.

Generated with Claude Opus 5.5.
"""
function plot_w_cpl!(ax, chain::WzChain; z=range(0, 4; length=400), color="#009E73", band_alpha=(0.15, 0.35),
                     bestfit::Bool=true, label=L"w_0w_a\mathrm{CDM}", linewidth=nothing)
    zs = collect(Float64, z)
    Q = w_quantiles(chain, zs; probs=[0.025, 0.16, 0.5, 0.84, 0.975])
    band!(ax, zs, Q[1, :], Q[5, :]; color=(color, band_alpha[1]))
    band!(ax, zs, Q[2, :], Q[4, :]; color=(color, band_alpha[2]))
    center = bestfit ? bestfit_w(chain, zs) : Q[3, :]
    line_kwargs = linewidth === nothing ? (;) : (; linewidth)
    lines!(ax, zs, center; color, label, line_kwargs...)
    return ax
end

"""
    plot_w_binned!(ax, chain; color="#0072B2", violin_color=(:gray, 0.35), violin_strokecolor=(:gray35, 0.8),
                   violin_width=0.12, show_violins=true, last_bin_center=nothing, last_bin_right=nothing,
                   markersize=nothing, linewidth=nothing, whiskerwidth=6, arrowsize=2whiskerwidth, arrow_weight=1.2,
                   violin_strokewidth=1, violin_tail=1e-4, label=L"\\mathrm{Binned}\\ w(z)")

Draw a binned-w(z) chain as in DESI DR2 Fig. 12:
- a marker at the posterior median of each bin amplitude, placed at the bin centre
- a horizontal bar spanning the bin
- a vertical bar spanning the 68% interval
- (optionally) a grey violin with the 1D posterior of the amplitude

The medians, intervals and violins all use the chain weights. Each violin is a weighted KDE with the weighted
Silverman bandwidth (`weighted_bandwidth`). It is cut at the weighted quantiles `violin_tail` and `1 - violin_tail`,
because many Nautilus points carry negligible weight and spread over the whole prior box.

The last bin has no upper edge. Its marker goes at `last_bin_center` (default: last edge + 1.05, i.e. z = 3.15 for
the DESI bins) and its horizontal bar reaches `last_bin_right` (default: mirrored about the centre), where it ends in
an arrowhead of height `arrowsize` (pixels) instead of a cap to show that the bin is open. Its arms are
`arrow_weight` times thicker than the bar, because anti-aliasing makes a 45° stroke look lighter than a vertical cap of
the same width (`arrow_weight=1` gives the exact bar width). The arrowhead tip overshoots `last_bin_right` by about
`linewidth / √2` pixels. For axes that stop at z = 2.5, as in `plotting.ipynb`, use e.g. `last_bin_center=2.3,
last_bin_right=2.47`, so that the tip stays clear of the frame.

`linewidth`, `whiskerwidth`, `markersize` and `violin_strokewidth` size the bars, markers and violin outlines (for
scaled publication themes).

Generated with Claude Opus 5.5.
"""
function plot_w_binned!(ax, chain::WzChain; color="#0072B2", violin_color=(:gray, 0.35),
                        violin_strokecolor=(:gray35, 0.8), violin_width=0.12, show_violins::Bool=true,
                        last_bin_center=nothing, last_bin_right=nothing, markersize=nothing,
                        linewidth=nothing, whiskerwidth=6, arrowsize=2whiskerwidth, arrow_weight=1.2,
                        violin_strokewidth=1, violin_tail=1e-4,
                        label=L"\mathrm{Binned}\ w(z)")
    bins = binned_w_summary(chain)
    centers = Float64[]
    lefts = Float64[]
    rights = Float64[]
    for bin in bins
        if isfinite(bin.z_hi)
            push!(centers, (bin.z_lo + bin.z_hi) / 2)
            push!(rights, bin.z_hi)
        else
            center = something(last_bin_center, bin.z_lo + 1.05)
            push!(centers, center)
            push!(rights, something(last_bin_right, 2center - bin.z_lo))
        end
        push!(lefts, bin.z_lo)
    end
    medians = [bin.median for bin in bins]

    if show_violins
        for (center, bin) in zip(centers, bins)
            lo, hi = weighted_quantile(bin.samples, bin.weights, (violin_tail, 1 - violin_tail))
            bandwidth = weighted_bandwidth(bin.samples, bin.weights)
            violin!(ax, fill(center, length(bin.samples)), bin.samples; weights=bin.weights, bandwidth,
                boundary=(lo - 4bandwidth, hi + 4bandwidth), datalimits=(lo, hi), width=violin_width,
                color=violin_color, strokecolor=violin_strokecolor, strokewidth=violin_strokewidth)
        end
    end
    bar_kwargs = linewidth === nothing ? (;) : (; linewidth)
    closed = [isfinite(bin.z_hi) for bin in bins]
    errorbars!(ax, centers[closed], medians[closed], (centers .- lefts)[closed], (rights .- centers)[closed];
        direction=:x, color, whiskerwidth, bar_kwargs...)
    # Open bin: a bar with a cap on the left and an arrowhead on the right, since it has no upper edge
    for i in findall(!, closed)
        shaft = lines!(ax, [lefts[i], rights[i]], [medians[i], medians[i]]; color, bar_kwargs...)
        t = Float64(shaft.linewidth[])
        scatter!(ax, [Point2d(lefts[i], medians[i])]; marker=_cap_marker(t, whiskerwidth), markersize=1,
            color, strokewidth=0)
        scatter!(ax, [Point2d(rights[i], medians[i])]; marker=_chevron_marker(arrow_weight * t, arrowsize), markersize=1,
            color, strokewidth=0)
    end
    errorbars!(ax, centers, medians, medians .- [bin.lo for bin in bins], [bin.hi for bin in bins] .- medians;
        direction=:y, color, whiskerwidth, bar_kwargs...)
    marker_kwargs = markersize === nothing ? (;) : (; markersize)
    scatter!(ax, centers, medians; color, label, marker_kwargs...)
    return ax
end

"""
    _cap_marker(t, w) -> BezierPath

End cap of a horizontal bar: a `t` × `w` rectangle centred on the anchor, in pixels (use with `markersize=1`), so it
matches the whiskers of `errorbars!` with `linewidth=t` and `whiskerwidth=w`.

Generated with Claude Opus 5.5.
"""
function _cap_marker(t, w)
    x, y = t / 2, w / 2
    return BezierPath([MoveTo(Point2d(-x, -y)), LineTo(Point2d(x, -y)), LineTo(Point2d(x, y)),
                       LineTo(Point2d(-x, y)), ClosePath()])
end

"""
    _chevron_marker(t, w) -> BezierPath

Open arrowhead pointing right, in pixels (use with `markersize=1`): two arms of thickness `t` that reach `w / 2`
above and below the bar, as tall as the whiskers. The mitred apex of the arm centre lines sits on the anchor, so a
bar of width `t` that ends at the anchor is covered by the arrowhead.

Generated with Claude Opus 5.5.
"""
function _chevron_marker(t, w)
    h = w / 2 # height of each arm
    l = h     # length of each arm along the bar (45° arms)
    α = atan(h, l)
    a = t / (2sin(α))        # apex offset of the outer and inner edges
    n = (t / 2) .* (sin(α), cos(α)) # half-thickness normal of the upper arm
    return BezierPath([MoveTo(Point2d(a, 0)),
                       LineTo(Point2d(-l + n[1], h + n[2])), LineTo(Point2d(-l - n[1], h - n[2])),
                       LineTo(Point2d(-a, 0)),
                       LineTo(Point2d(-l - n[1], -h + n[2])), LineTo(Point2d(-l + n[1], -h - n[2])),
                       ClosePath()])
end

"""
    figure_wz_fig12(cpl, binned; zmax=4.0, ylims=(-2.2, -0.55), size=(600, 480)) -> Figure

Stand-alone replica of DESI DR2 II Fig. 12: CPL bands and best fit, binned amplitudes with violins, and the ΛCDM
line w = -1.

Generated with Claude Opus 5.5.
"""
function figure_wz_fig12(cpl::WzChain, binned::WzChain; zmax=4.0, ylims=(-2.2, -0.55), size=(600, 480))
    fig = Figure(; size)
    ax = Axis(fig[1, 1]; xlabel=L"z", ylabel=L"w(z)", limits=((0, zmax), ylims))
    hlines!(ax, [-1.0]; color=:gray50, linestyle=:dash)
    plot_w_cpl!(ax, cpl; z=range(0, zmax; length=400))
    plot_w_binned!(ax, binned)
    patchsize = (32, 20)
    cpl_handle = LineElement(; color="#009E73")
    axislegend(ax, [cpl_handle, errorbar_legend_handle("#0072B2"; patchsize)],
        [L"w_0w_a\mathrm{CDM}", L"\mathrm{Binned}\ w(z)"]; position=:rt, patchsize)
    return fig
end

"""
    errorbar_legend_handle(color; linewidth=1.5, markersize=8, whiskerwidth=7, patchsize=(32, 20)) -> Vector{LegendElement}

Legend glyph for a point with x and y error bars, as in the DESI DR2 Fig. 12 legend: a horizontal and a vertical bar with
end caps, and a dot at the centre. Makie has no errorbar legend glyph, so the bars and caps are one `LineElement`
whose `linepoints` (fractions of the legend patch) are separated by NaN to break the line.

`linewidth`, `markersize` and `whiskerwidth` are in pixels, like the same keywords of `plot_w_binned!`, so pass the
plotted values to match the bars. `patchsize` must equal the legend's `patchsize` (a keyword of `axislegend`, or the
`Legend` theme) so that the caps are converted from pixels to patch fractions correctly.

Generated with Claude Sonnet 5.5.
"""
function errorbar_legend_handle(color; linewidth=1.5, markersize=8, whiskerwidth=7, patchsize=(32, 20))
    capx = whiskerwidth / 2 / patchsize[1] # half-length of the caps of the vertical bar, as a patch fraction
    capy = whiskerwidth / 2 / patchsize[2] # same for the caps of the horizontal bar
    nan = Point2f(NaN, NaN)
    h = 0.5 # centre of the patch
    v0, v1 = 0.1, 0.9 # the vertical bar is a little shorter than the horizontal one
    points = [Point2f(0, h), Point2f(1, h), nan,
              Point2f(0, h - capy), Point2f(0, h + capy), nan,
              Point2f(1, h - capy), Point2f(1, h + capy), nan,
              Point2f(h, v0), Point2f(h, v1), nan,
              Point2f(h - capx, v0), Point2f(h + capx, v0), nan,
              Point2f(h - capx, v1), Point2f(h + capx, v1)]
    bars = LineElement(; color, linewidth, linepoints=points)
    dot = MarkerElement(; color, marker=:circle, markersize)
    return [bars, dot]
end

end # module
