"""Nautilus posterior of a w(z) model (lcdm, cpl, binned) with this project's data, written as an HDF5 chain that
the Julia `load_wz_chain` reads (binning/plot_fig12.jl, plotting.ipynb).

    cd binning/python
    uv run python run_nautilus.py --model cpl
    uv run python run_nautilus.py --model binned
    uv run python run_nautilus.py --model cpl --likelihoods DESI_BAO,Planck_Zhai,PantheonPlus_Marg --label validation

Nautilus checkpoints to <output-root>/nautilus/<label_><model>.hdf5; rerun with --resume to continue. The chain goes
to <output-root>/<label_><model>_<UTCstamp>.h5.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"  # one BLAS thread per process; the pool provides the parallelism

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import nautilus
from scipy.optimize import minimize

from wz.data import DEFAULT_LIKELIHOODS, SUPPORTED_LIKELIHOODS
from wz.io_julia import save_wz_chain
from wz.likelihood import LogLike, get_evaluator
from wz.stats import print_summary

CHAINS_DIR = Path(__file__).resolve().parent.parent / "chains"


def parse_args(argv=None):
    """Command-line options.

    Generated with Claude Opus 5.5.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["lcdm", "cpl", "binned"])
    p.add_argument("--likelihoods", default=",".join(DEFAULT_LIKELIHOODS),
                   help=f"comma-separated subset of {', '.join(SUPPORTED_LIKELIHOODS)}")
    p.add_argument("--edges", default="0.7,1.4,2.1", help="bin edges of the binned model")
    p.add_argument("--smoothing", type=float, default=0.02, help="tanh width s of the binned model")
    p.add_argument("--n-live", type=int, default=3000)
    p.add_argument("--n-eff", type=float, default=10000)
    p.add_argument("--pool", type=int, default=16, help="worker processes for likelihood calls (1: serial)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--label", default="", help="prefix of the output names")
    p.add_argument("--output-root", default=str(CHAINS_DIR))
    p.add_argument("--resume", action="store_true", help="continue from the Nautilus checkpoint")
    p.add_argument("--overwrite", action="store_true", help="discard an existing checkpoint and start over")
    return p.parse_args(argv)


def polish(ev, x0, scale):
    """Maximum of the likelihood inside the prior box near `x0`: Powell, then Nelder-Mead with a simplex scaled
    by `scale`, then Powell again. Returns (x, χ²).

    Generated with Claude Opus 5.5.
    """
    f = ev.total_chi2
    bounds = [tuple(b) for b in ev.bounds]
    best_x, best_f = np.asarray(x0, dtype=float), f(x0)
    for method in ("Powell", "Nelder-Mead", "Powell"):
        if method == "Powell":
            options = dict(xtol=1e-8, ftol=1e-12, maxfev=40000)
        else:
            simplex = [best_x] + [np.clip(best_x + np.eye(best_x.size)[i] * scale[i], ev.bounds[:, 0],
                                          ev.bounds[:, 1]) for i in range(best_x.size)]
            options = dict(xatol=1e-9, fatol=1e-10, maxfev=40000, adaptive=True, initial_simplex=np.array(simplex))
        r = minimize(f, best_x, method=method, bounds=bounds, options=options)
        if r.fun < best_f:
            best_x, best_f = np.asarray(r.x, dtype=float), float(r.fun)
    return best_x, best_f


def main(argv=None):
    """Run Nautilus (exploration phase discarded), find the MAP and the ΛCDM best fit, write the Julia chain and
    print a summary.

    Generated with Claude Opus 5.5.
    """
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    likelihoods = tuple(s.strip() for s in args.likelihoods.split(",") if s.strip())
    edges = tuple(float(e) for e in args.edges.split(","))
    ev = get_evaluator(args.model, likelihoods, edges, args.smoothing)
    names = ev.param_names
    prefix = f"{args.label}_{args.model}" if args.label else args.model
    out_root = Path(args.output_root)
    checkpoint = out_root / "nautilus" / f"{prefix}.hdf5"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists() and not (args.resume or args.overwrite):
        sys.exit(f"checkpoint {checkpoint} exists; pass --resume to continue it, --overwrite to start over, "
                 "or use another --label")
    if checkpoint.exists() and args.overwrite:
        checkpoint.unlink()

    started = datetime.now(timezone.utc)
    print(f"model {args.model}  likelihoods {','.join(likelihoods)}  parameters {', '.join(names)}")
    if args.model == "binned":
        print(f"edges {edges}  smoothing {args.smoothing}")
    print(f"nautilus {nautilus.__version__}  n_live {args.n_live}  n_eff {args.n_eff}  pool {args.pool}  "
          f"seed {args.seed}  checkpoint {checkpoint}")
    t0 = time.perf_counter()
    x_test = ev.bounds.mean(axis=1)
    ev.loglike(x_test)
    t_eval = time.perf_counter()
    for _ in range(20):
        ev.loglike(x_test)
    print(f"likelihood cost {(time.perf_counter() - t_eval) / 20 * 1e3:.2f} ms per call (serial)")

    prior = nautilus.Prior()
    for name, (lo, hi) in zip(names, ev.bounds):
        prior.add_parameter(name, dist=(float(lo), float(hi)))
    sampler = nautilus.Sampler(prior, LogLike(args.model, likelihoods, edges, args.smoothing), n_live=args.n_live,
                               pool=args.pool if args.pool > 1 else None, pass_dict=False, seed=args.seed,
                               filepath=str(checkpoint), resume=args.resume)
    # The exploration phase samples the prior-sized bound inefficiently; dropping it leaves the posterior unbiased.
    sampler.run(n_eff=args.n_eff, discard_exploration=True, verbose=True)
    t_sample = time.perf_counter() - t0

    points, log_w, log_l = sampler.posterior()
    eq_points, _, eq_log_l = sampler.posterior(equal_weight=True)
    n_eff, n_like = float(sampler.n_eff), int(sampler.n_like)
    print(f"\nsampling done in {t_sample / 60:.1f} min: {n_like} likelihood calls, n_eff {n_eff:.0f}, "
          f"{len(eq_points)} equal-weight draws")

    scale = 0.5 * eq_points.std(axis=0, ddof=1)
    start = points[np.argmax(log_l)]
    map_x, map_chi2 = polish(ev, start, scale)
    print(f"MAP: best sample χ² {-2 * log_l.max():.4f} → polished {map_chi2:.4f}")
    map_params = ev.params_from_vector(map_x)
    map_comps = ev.chi2_components(map_params)

    if args.model == "lcdm":
        lcdm_chi2 = map_chi2
    else:
        ev_l = get_evaluator("lcdm", likelihoods)
        lcdm_chi2 = np.inf
        for x0 in (map_x, eq_points.mean(axis=0)):
            x0_l = np.array([dict(zip(names, x0))[n] for n in ev_l.param_names])
            _, chi2_l = polish(ev_l, x0_l, scale[:len(ev_l.param_names)])
            lcdm_chi2 = min(lcdm_chi2, chi2_l)
        print(f"ΛCDM best fit χ² {lcdm_chi2:.4f}")

    finished = datetime.now(timezone.utc)
    stamp = finished.strftime("%Y%m%dT%H%M%SZ")
    out_path = out_root / f"{prefix}_{stamp}.h5"
    attributes = {
        "sampler": f"nautilus {nautilus.__version__}",
        "seed": args.seed, "n_live": args.n_live, "n_eff": n_eff, "n_like_calls": n_like,
        "lcdm_map_chi2": float(lcdm_chi2), "map_chi2": float(map_chi2),
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "wall_time_s": time.perf_counter() - t0,
    }
    save_wz_chain(out_path, model=ev.model, param_names=names, samples=eq_points, logpost=eq_log_l,
                  map_params=map_params, map_chi2_components=map_comps, likelihoods=likelihoods,
                  attributes=attributes, weighted={"points": points, "log_w": log_w, "log_l": log_l})

    print_summary(names, points, map_x, np.exp(log_w - log_w.max()))
    print("\nχ² at the MAP:")
    for name, value in map_comps.items():
        print(f"  {name:20s} {value:12.4f}")
    print(f"  {'total':20s} {map_chi2:12.4f}   Δχ² vs ΛCDM ({lcdm_chi2:.4f}): {map_chi2 - lcdm_chi2:+.4f}")
    print(f"n_eff {n_eff:.0f}, likelihood calls {n_like}, wall time {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
