# Replica of DESI DR2 II Fig. 12 from the WzBinning chains.
#
#   julia --project=. binning/plot_fig12.jl                       # newest cpl_*.h5 and binned_*.h5 in binning/chains
#   julia --project=. binning/plot_fig12.jl CPL.h5 BINNED.h5 [OUTPUT_STEM]
#
# Writes OUTPUT_STEM.pdf and OUTPUT_STEM.png (default binning/figures/wz_fig12).

include(joinpath(@__DIR__, "WzBinning.jl"))
include(joinpath(@__DIR__, "WzPlots.jl"))
using .WzBinning, .WzPlots
using CairoMakie

"""
    newest_chain(prefix) -> String

Most recent `<prefix>_*.h5` in `binning/chains` (file names carry a UTC timestamp, so the lexical maximum is the
newest).

Generated with Claude Opus 5.5.
"""
function newest_chain(prefix::String)
    dir = joinpath(@__DIR__, "chains")
    files = filter(f -> startswith(f, prefix * "_") && endswith(f, ".h5"), readdir(dir))
    isempty(files) && error("no $(prefix)_*.h5 chain in $dir; run binning/python/run_nautilus.py first")
    return joinpath(dir, maximum(files))
end

cpl_path = length(ARGS) >= 1 ? ARGS[1] : newest_chain("cpl")
binned_path = length(ARGS) >= 2 ? ARGS[2] : newest_chain("binned")
stem = length(ARGS) >= 3 ? ARGS[3] : joinpath(@__DIR__, "figures", "wz_fig12")

cpl = load_wz_chain(cpl_path)
binned = load_wz_chain(binned_path)
println(cpl, "\n", binned)

fig = figure_wz_fig12(cpl, binned)
mkpath(dirname(stem))
save(stem * ".pdf", fig)
save(stem * ".png", fig; px_per_unit=2)
println("saved ", stem, ".pdf and .png")
