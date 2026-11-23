"""χ² of each data set (algebra of src/likelihoods.jl) and the log-likelihood handed to Nautilus.

`Evaluator` holds the data and the background grid for one model and one data combination. `LogLike` is the
picklable callable given to the sampler: it carries only the configuration and builds its `Evaluator` lazily, once
per process, so worker processes never receive the data through pickling whatever the start method.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg.blas import dtrmv

from . import data as wzdata
from .background import C_KMS, BackgroundGrid
from .models import COSMO_DEFAULTS, model_from_tag, param_bounds, param_names


def _lower_quadform(inv_chol, r):
    """|L⁻¹ r|² with the triangular BLAS product (half the flops of a full matrix-vector product).

    Generated with Claude Opus 5.5.
    """
    u = dtrmv(inv_chol, r, lower=1)
    return float(u @ u), u


class Evaluator:
    """χ² components for one model (`tag`, `edges`, `smoothing`) and one data combination.

    Generated with Claude Opus 5.5.
    """

    def __init__(self, tag, likelihoods=wzdata.DEFAULT_LIKELIHOODS, edges=(0.7, 1.4, 2.1), smoothing=0.02):
        self.model = model_from_tag(tag, edges, smoothing)
        self.likelihood_names = tuple(likelihoods)
        self.likes = wzdata.load_likelihoods(self.likelihood_names)
        self.param_names = param_names(self.model, self.likelihood_names)
        self.bounds = np.array(param_bounds(self.model, self.likelihood_names))
        self.n_w = len(self.model.w_names)

        z_dm, z_growth = [], []
        self._dm_slices = {}
        for like in self.likes:
            if isinstance(like, (wzdata.PantheonPlusSH0ES, wzdata.PantheonPlusMarg, wzdata.DESIBAO)):
                z = like.z_cmb if hasattr(like, "z_cmb") else like.z
                start = sum(len(v) for v in z_dm)
                self._dm_slices[like.name] = slice(start, start + z.size)
                z_dm.append(z)
            if isinstance(like, wzdata.GrowthRSD):
                z_growth.append(like.z)
                start = sum(len(v) for v in z_dm)   # D_C there too, for the fiducial-cosmology correction
                self._dm_slices[like.name] = slice(start, start + like.z.size)
                z_dm.append(like.z)
        z_dm = np.concatenate(z_dm) if z_dm else np.zeros(0)
        # The grid deduplicates breakpoints; map every requested redshift onto its breakpoint.
        self.grid = BackgroundGrid(self.model, z_dm=z_dm, z_growth=np.concatenate(z_growth) if z_growth else ())
        self._desi = next((l for l in self.likes if isinstance(l, wzdata.DESIBAO)), None)
        if self._desi is not None:
            self._desi_lnf = self.model.lnf_terms(np.log1p(self._desi.z))

    def chi2_components(self, params):
        """χ² of every likelihood at `params` (a dict by name; missing cosmological parameters take
        `COSMO_DEFAULTS`), or None when the point is unphysical (Ω_DE ≤ 0, CPL with w0 + wa ≥ 0, non-finite
        background).

        Generated with Claude Opus 5.5.
        """
        value = lambda name: float(params.get(name, COSMO_DEFAULTS[name]))
        theta_w = np.array([params[name] for name in self.model.w_names], dtype=float)
        if not self.model.constraint_ok(theta_w):
            return None
        Om0, h, Ob0 = value("Om0"), value("h"), value("Ob0")
        M, sigma8 = value("M_nuisance"), value("sigma8_0")
        bg = self.grid.solve(theta_w, Om0, h, Ob0)
        if bg is None:
            return None
        chi_req = self.grid.chi_requested(bg)
        dh0 = C_KMS / bg.H0
        out = {}
        for like in self.likes:
            if isinstance(like, wzdata.PantheonPlusSH0ES):
                dm = dh0 * chi_req[self._dm_slices[like.name]]
                with np.errstate(divide="ignore", invalid="ignore"):
                    mu = np.where(like.is_calibrator, like.mu_cepheid,
                                  5 * np.log10(like.zhel_plus_one * dm) + 25)
                out[like.name], _ = _lower_quadform(like.inv_chol, like.mb_obs - (mu + M))
            elif isinstance(like, wzdata.PantheonPlusMarg):
                dm = dh0 * chi_req[self._dm_slices[like.name]]
                r = like.mb_obs - (5 * np.log10(like.zhel_plus_one * dm) + 25)
                A, u = _lower_quadform(like.inv_chol, r)
                B = float(like.inv_chol_ones @ u)
                C = float(like.inv_chol_ones @ like.inv_chol_ones)
                out[like.name] = A + np.log(C / (2 * np.pi)) - B * B / C
            elif isinstance(like, wzdata.DESIBAO):
                rd = bg.sound_horizon(bg.z_drag)[0]
                dm = dh0 * chi_req[self._dm_slices[like.name]]
                c, Bm = self._desi_lnf
                x = np.log1p(like.z)
                dh = dh0 / np.sqrt(bg.E2_x(x, c + Bm @ theta_w))
                theory = np.select([like.flag == 1, like.flag == 2, like.flag == 3],
                                   [np.cbrt(like.z * dm * dm * dh) / rd, dm / dh, dm / rd], dh / rd)
                u = like.inv_chol @ (like.value - theory)
                out[like.name] = float(u @ u)
            elif isinstance(like, wzdata.PlanckCompressed):
                zs = bg.z_star
                chi_s = bg.chi(zs)[0]
                rs = bg.sound_horizon(zs)[0]
                R = np.sqrt(Om0) * chi_s
                la = np.pi * dh0 * chi_s / rs
                u = like.inv_chol @ (np.array([R, la, bg.omega_b]) - like.obs)
                out[like.name] = float(u @ u)
            elif isinstance(like, wzdata.GrowthRSD):
                # Fiducial-cosmology correction: the prediction is divided by H D_C / (H_fid D_C,fid) = E χ / (E D_C)_fid
                chi_g = chi_req[self._dm_slices[like.name]]
                ratio = np.sqrt(bg.E2_x(np.log1p(like.z))) * chi_g / like.EDc_fid
                r = like.fs8_obs - bg.fs8(sigma8) / ratio
                total = float(np.sum((r[like.idx_diag] / like.fs8_err[like.idx_diag]) ** 2))
                u = like.inv_chol_wig @ r[like.idx_wig]
                v = like.inv_chol_boss @ r[like.idx_boss]
                out[like.name] = total + float(u @ u) + float(v @ v)
        return out

    def params_from_vector(self, theta):
        """Dict of the sampled parameters from a flat vector in `param_names` order.

        Generated with Claude Opus 5.5.
        """
        return dict(zip(self.param_names, map(float, theta)))

    def total_chi2(self, theta):
        """Sum of the χ² components at the flat vector `theta`; inf outside the prior box, for unphysical points
        and when any term is not finite.

        Generated with Claude Opus 5.5.
        """
        theta = np.asarray(theta, dtype=float)
        if np.any(theta < self.bounds[:, 0]) or np.any(theta > self.bounds[:, 1]):
            return np.inf
        try:
            comps = self.chi2_components(self.params_from_vector(theta))
        except (FloatingPointError, ValueError, ZeroDivisionError, OverflowError):
            return np.inf
        if comps is None:
            return np.inf
        total = sum(comps.values())
        return total if np.isfinite(total) else np.inf

    def loglike(self, theta):
        """-χ²/2 at the flat vector `theta` (-inf for rejected points).

        Generated with Claude Opus 5.5.
        """
        return -0.5 * self.total_chi2(theta)


@lru_cache(maxsize=None)
def get_evaluator(tag, likelihoods=wzdata.DEFAULT_LIKELIHOODS, edges=(0.7, 1.4, 2.1), smoothing=0.02):
    """One `Evaluator` per configuration and process.

    Generated with Claude Opus 5.5.
    """
    return Evaluator(tag, tuple(likelihoods), tuple(edges), smoothing)


def chi2_components(params, tag, likelihoods=wzdata.DEFAULT_LIKELIHOODS, edges=(0.7, 1.4, 2.1), smoothing=0.02):
    """χ² by likelihood name for a parameter dict (None when the point is rejected).

    Generated with Claude Opus 5.5.
    """
    return get_evaluator(tag, tuple(likelihoods), tuple(edges), float(smoothing)).chi2_components(params)


def loglike(params, tag, likelihoods=wzdata.DEFAULT_LIKELIHOODS, edges=(0.7, 1.4, 2.1), smoothing=0.02):
    """-χ²_total/2 for a parameter dict, -inf when rejected.

    Generated with Claude Opus 5.5.
    """
    ev = get_evaluator(tag, tuple(likelihoods), tuple(edges), float(smoothing))
    return ev.loglike([params[n] for n in ev.param_names])


class LogLike:
    """Picklable log-likelihood of a flat parameter vector for Nautilus. Only the configuration is pickled; the
    `Evaluator` (data, grids) is built on first use in each process.

    Generated with Claude Opus 5.5.
    """

    def __init__(self, tag, likelihoods, edges=(0.7, 1.4, 2.1), smoothing=0.02):
        self.config = (tag, tuple(likelihoods), tuple(float(e) for e in edges), float(smoothing))

    def __call__(self, theta):
        return get_evaluator(*self.config).loglike(theta)
