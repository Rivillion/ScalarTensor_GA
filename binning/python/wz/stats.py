"""Posterior statistics from weighted points (the `nautilus/` group of a chain file) or, with `weights=None`, from
equal-weight samples (the `samples` dataset). The definitions match the weighted statistics of binning/WzBinning.jl.
"""

import numpy as np

from .models import model_from_tag


def weighted_quantile(x, weights, probs):
    """Quantiles of `x` with `weights`: each point stands at the midpoint of its weight in the cumulative
    distribution, F_k = (Σ_{j≤k} w_j - w_k/2) / Σ w, and the quantile is linearly interpolated in F (the Python twin
    of `weighted_quantile` in binning/WzBinning.jl; Hazen's definition for uniform weights).

    Generated with Claude Opus 5.5.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cdf = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return np.interp(probs, cdf, xs)


def weighted_std(x, weights):
    """Weighted standard deviation, Σw(x - μ)² / (Σw - Σw²/Σw); the n - 1 estimator for uniform weights.

    Generated with Claude Opus 5.5.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(weights, dtype=float)
    mu = np.average(x, weights=w)
    total = w.sum()
    return float(np.sqrt(np.sum(w * (x - mu) ** 2) / (total - np.sum(w * w) / total)))


def effective_sample_size(weights):
    """Kish effective sample size (Σw)² / Σw².

    Generated with Claude Opus 5.5.
    """
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w * w))


def param_table(names, samples, map_x=None, weights=None):
    """One dict per parameter: mean, std, median, 68% and 95% equal-tailed intervals and (optionally) the MAP.
    `weights=None` treats the samples as equal-weight.

    Generated with Claude Opus 5.5.
    """
    w = np.ones(len(samples)) if weights is None else np.asarray(weights, dtype=float)
    rows = []
    for k, name in enumerate(names):
        x = samples[:, k]
        q = weighted_quantile(x, w, [0.025, 0.16, 0.5, 0.84, 0.975])
        rows.append(dict(name=name, mean=np.average(x, weights=w), std=weighted_std(x, w), median=q[2],
                         lo68=q[1], hi68=q[3], lo95=q[0], hi95=q[4], map=np.nan if map_x is None else map_x[k]))
    return rows


def print_summary(names, samples, map_x=None, weights=None):
    """Print `param_table` as a fixed-width table.

    Generated with Claude Opus 5.5.
    """
    print(f"\n{'parameter':12s} {'mean':>10s} {'std':>9s} {'median':>10s} {'68% interval':>23s} "
          f"{'95% interval':>23s} {'MAP':>10s}")
    for r in param_table(names, samples, map_x, weights):
        print(f"{r['name']:12s} {r['mean']:10.5f} {r['std']:9.5f} {r['median']:10.5f} "
              f"[{r['lo68']:10.5f},{r['hi68']:10.5f}] [{r['lo95']:10.5f},{r['hi95']:10.5f}] {r['map']:10.5f}")


def correlation(names, samples, weights=None):
    """Posterior correlation matrix as (names, matrix).

    Generated with Claude Opus 5.5.
    """
    C = np.cov(samples, rowvar=False, aweights=weights)
    s = np.sqrt(np.diag(C))
    return list(names), C / np.outer(s, s)


def w_quantiles(chain, z, probs=(0.025, 0.16, 0.5, 0.84, 0.975)):
    """Point-by-point posterior quantiles of w(z) for a chain dict from `io_julia.load_wz_chain`: one row per
    probability, one column per redshift (the Python twin of `w_quantiles` in binning/WzBinning.jl). Uses the
    weighted points when the chain has them.

    Generated with Claude Opus 5.5.
    """
    attrs = chain["attrs"]
    model = model_from_tag(attrs["model"], attrs.get("edges", (0.7, 1.4, 2.1)), attrs.get("smoothing", 0.02))
    cols = [chain["param_names"].index(n) for n in model.w_names]
    points = chain.get("points", chain["samples"])
    weights = chain.get("weights", np.ones(len(points)))
    c, B = model.w_terms(np.asarray(z, dtype=float))
    W = c[None, :] + points[:, cols] @ B.T
    return np.array([weighted_quantile(W[:, j], weights, probs) for j in range(W.shape[1])]).T
