"""HDF5 writer in the format read by `load_wz_chain` (binning/WzBinning.jl).

h5py writes row-major and HDF5.jl reads column-major, so every dataset appears with reversed dimensions in Julia:
`samples` is written as (1, n_param, n_draw) and read as draws × params × 1 chain, `logpost` as (1, n_draw).
"""

import os
from pathlib import Path

import h5py
import numpy as np

_STR = h5py.string_dtype("utf-8")


def save_wz_chain(path, *, model, param_names, samples, logpost, map_params, map_chi2_components, likelihoods,
                  attributes=None, weighted=None):
    """Write a chain atomically (to `<path>.tmp`, then renamed).

    - `samples`: draws × params (equal-weight posterior), `logpost`: one value per draw
    - `map_params`, `map_chi2_components`: dicts by name
    - `attributes`: root attributes (str, numbers, arrays)
    - `weighted`: optional dict with the unequal-weight Nautilus `points`, `log_w`, `log_l`, stored in group
      `nautilus/` (ignored by Julia)

    Generated with Claude Opus 5.5.
    """
    samples = np.asarray(samples, dtype=float)
    logpost = np.asarray(logpost, dtype=float)
    if samples.shape != (logpost.size, len(param_names)):
        raise ValueError(f"samples {samples.shape} do not match {logpost.size} draws × {len(param_names)} params")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with h5py.File(tmp, "w") as f:
        f["samples"] = samples.T[None, :, :]
        f["logpost"] = logpost[None, :]
        f.create_dataset("param_names", data=list(param_names), dtype=_STR)
        f.create_dataset("likelihoods", data=list(likelihoods), dtype=_STR)
        names = list(map_params)
        f.create_dataset("map/names", data=names, dtype=_STR)
        f["map/values"] = np.array([map_params[n] for n in names], dtype=float)
        comps = list(map_chi2_components)
        f.create_dataset("map/chi2_names", data=comps, dtype=_STR)
        f["map/chi2_values"] = np.array([map_chi2_components[n] for n in comps], dtype=float)
        if weighted is not None:
            for key in ("points", "log_w", "log_l"):
                f[f"nautilus/{key}"] = np.asarray(weighted[key], dtype=float)
        f.attrs["model"] = model.tag
        if model.tag == "binned":
            f.attrs["edges"] = np.asarray(model.edges, dtype=float)
            f.attrs["smoothing"] = float(model.smoothing)
        for key, value in (attributes or {}).items():
            f.attrs[key] = value
    os.replace(tmp, path)
    return path


def load_wz_chain(path):
    """Read a chain written by `save_wz_chain` back into a dict (for checks; Julia has its own reader). When the
    file has the `nautilus/` group, the dict also holds the weighted `points` with `weights` (exp(log_w), normalised
    to sum 1) and `log_l`.

    Generated with Claude Opus 5.5.
    """
    with h5py.File(path, "r") as f:
        decode = lambda d: [s.decode() if isinstance(s, bytes) else s for s in d[()]]
        chain = {
            "samples": f["samples"][0].T,
            "logpost": f["logpost"][0],
            "param_names": decode(f["param_names"]),
            "likelihoods": decode(f["likelihoods"]),
            "map": dict(zip(decode(f["map/names"]), f["map/values"][()])),
            "map_chi2": dict(zip(decode(f["map/chi2_names"]), f["map/chi2_values"][()])),
            "attrs": dict(f.attrs),
        }
        if "nautilus" in f:
            log_w = f["nautilus/log_w"][()]
            weights = np.exp(log_w - log_w.max())
            chain["points"] = f["nautilus/points"][()]
            chain["weights"] = weights / weights.sum()
            chain["log_l"] = f["nautilus/log_l"][()]
        return chain
