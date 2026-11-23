"""Posterior statistics of chain files written by run_nautilus.py.

    cd binning/python
    uv run python summarize_chain.py ../chains/cpl_<stamp>.h5 [more.h5 ...] [--corr]
"""

import argparse

import numpy as np

from wz.io_julia import load_wz_chain
from wz.stats import correlation, print_summary


def main(argv=None):
    """Print the parameter table, MAP χ² components and (with --corr) the correlation matrix of each file.

    Generated with Claude Opus 5.5.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+")
    p.add_argument("--corr", action="store_true", help="also print the correlation matrix")
    args = p.parse_args(argv)
    for path in args.paths:
        c = load_wz_chain(path)
        a = c["attrs"]
        names = c["param_names"]
        points, weights = c.get("points", c["samples"]), c.get("weights")
        kind = "weighted points" if weights is not None else "equal-weight draws (no nautilus/ group)"
        print(f"\n{path}\nmodel {a['model']}  likelihoods {','.join(c['likelihoods'])}  "
              f"{points.shape[0]} {kind}  n_eff {a.get('n_eff', np.nan):.0f}")
        print_summary(names, points, [c["map"][n] for n in names], weights)
        total = sum(c["map_chi2"].values())
        print("χ² at the MAP: " + ", ".join(f"{k} {v:.4f}" for k, v in c["map_chi2"].items())
              + f"; total {total:.4f}, Δχ² vs ΛCDM {total - a.get('lcdm_map_chi2', np.nan):+.4f}")
        if args.corr:
            _, C = correlation(names, points, weights)
            print(" " * 12 + "".join(f"{n:>11s}" for n in names))
            for n, row in zip(names, C):
                print(f"{n:12s}" + "".join(f"{v:11.3f}" for v in row))


if __name__ == "__main__":
    main()
