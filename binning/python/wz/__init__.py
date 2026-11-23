"""Nautilus posteriors of binned and CPL w(z) with this project's data (DESI DR2 II Fig. 12 replica).

The physics and data handling follow the GA conventions (src/likelihoods.jl, src/background_GPTv2.jl);
tests/test_crosscheck.py checks every χ² component against the frozen reference tests/reference_chi2.toml.
Chains are plotted in Julia with binning/WzBinning.jl and binning/WzPlots.jl.
"""
