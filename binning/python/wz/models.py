"""Dark-energy equation-of-state models and the parameter bookkeeping of
`build_parameter_spec` in src/minimize_GPTv2.jl.

Every model is linear in its parameters θw, both in w(z) and in ln f_DE(x) = ln[ρ_DE(x)/ρ_DE(0)], where
x = ln(1+z) = -N. The models therefore expose `w_terms(z)` and `lnf_terms(x)`, which return (c, B) with
w = c + B @ θw and ln f_DE = c + B @ θw. The background precomputes these terms once on its fixed grids, so
a likelihood call costs a few matrix-vector products.
"""

import numpy as np
from scipy.special import expit

Z_MAX = 1.0e5
X_MAX = np.log1p(Z_MAX)

COSMO_PRIORS = {
    "Om0": (0.05, 0.7),
    "h": (0.4, 1.0),
    "M_nuisance": (-21.0, -18.0),
    "sigma8_0": (0.4, 1.2),
    "Ob0": (0.01, 0.15),
}
COSMO_DEFAULTS = {"Om0": 0.295, "h": 0.685, "Ob0": 0.0475, "sigma8_0": 0.80, "M_nuisance": -19.4}

_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


class LCDMModel:
    """Cosmological constant, w = -1, with no free parameters.

    Generated with Claude Opus 5.5.
    """

    tag = "lcdm"
    w_names: list[str] = []
    w_bounds: list[tuple[float, float]] = []
    transitions: np.ndarray = np.zeros(0)

    def constraint_ok(self, theta_w):
        """Non-box prior conditions; none for ΛCDM.

        Generated with Claude Opus 5.5.
        """
        return True

    def w_terms(self, z):
        """(c, B) with w(z) = c + B @ θw.

        Generated with Claude Opus 5.5.
        """
        z = np.asarray(z, dtype=float)
        return np.full(z.shape, -1.0), np.zeros(z.shape + (0,))

    def lnf_terms(self, x):
        """(c, B) with ln f_DE(x) = c + B @ θw; ρ_DE is constant.

        Generated with Claude Opus 5.5.
        """
        x = np.asarray(x, dtype=float)
        return np.zeros(x.shape), np.zeros(x.shape + (0,))

    def w_of_z(self, theta_w, z):
        """Equation of state at redshift `z`.

        Generated with Claude Opus 5.5.
        """
        c, B = self.w_terms(z)
        return c + B @ np.asarray(theta_w, dtype=float)


class CPLModel(LCDMModel):
    """CPL parametrization, w(z) = w0 + wa z/(1+z), with w0 + wa < 0.

    Generated with Claude Opus 5.5.
    """

    tag = "cpl"
    w_names = ["w0", "wa"]
    w_bounds = [(-3.0, 1.0), (-3.0, 2.0)]

    def constraint_ok(self, theta_w):
        """DESI's condition w0 + wa < 0 (matter domination at high redshift).

        Generated with Claude Opus 5.5.
        """
        return theta_w[0] + theta_w[1] < 0

    def w_terms(self, z):
        """(c, B) with w(z) = c + B @ θw.

        Generated with Claude Opus 5.5.
        """
        z = np.asarray(z, dtype=float)
        return np.zeros(z.shape), np.stack([np.ones(z.shape), z / (1 + z)], axis=-1)

    def lnf_terms(self, x):
        """ln f_DE = 3(1 + w0 + wa) x - 3 wa (1 - e^{-x}) up to z = 1e5, and constant w = w(1e5) above it,
        as the former Julia core did.

        Generated with Claude Opus 5.5.
        """
        x = np.asarray(x, dtype=float)
        xc = np.minimum(x, X_MAX)
        above = np.maximum(x - X_MAX, 0.0)
        b_wa = 3 * xc - 3 * (-np.expm1(-xc)) + 3 * (Z_MAX / (1 + Z_MAX)) * above
        return 3 * x, np.stack([3 * x, b_wa], axis=-1)


class BinnedWModel(LCDMModel):
    """Binned equation of state with tanh-smoothed transitions (Lodha et al. 2025, Eq. 13):
    w(z) = w_1 + Σ_i (w_{i+1} - w_i)/2 [1 + tanh((z - z_i)/s)].

    ln f_DE = 3(1 + w_1) x + (3/2) Σ_i (w_{i+1} - w_i) G_i(x) with the parameter-free integrals
    G_i(x) = ∫_0^x [1 + tanh((e^{x'} - 1 - z_i)/s)] dx', tabulated once on a fine grid and evaluated with cubic
    Hermite interpolation (the derivative is known exactly). Above z_i + 40 s the tanh is saturated and G_i grows
    linearly with slope 2.

    Generated with Claude Opus 5.5.
    """

    tag = "binned"

    def __init__(self, edges=(0.7, 1.4, 2.1), smoothing=0.02, table_step=2.0e-4):
        edges = np.asarray(edges, dtype=float)
        if edges.size == 0 or not np.all(np.isfinite(edges)) or edges[0] <= 0 or np.any(np.diff(edges) <= 0):
            raise ValueError("bin edges must be finite, positive and strictly increasing")
        if not (np.isfinite(smoothing) and smoothing > 0):
            raise ValueError("smoothing must be finite and positive")
        self.edges = edges
        self.smoothing = float(smoothing)
        self.transitions = edges
        self.w_names = [f"w[{i}]" for i in range(1, edges.size + 2)]
        self.w_bounds = [(-3.0, 1.0)] * (edges.size + 1)
        self._tables = [self._tabulate(z_i, table_step) for z_i in edges]

    def _dG(self, x, z_i):
        """dG_i/dx = 1 + tanh(u) = 2 expit(2u), u = (e^x - 1 - z_i)/s, without cancellation at u << 0.

        Generated with Claude Opus 5.5.
        """
        return 2.0 * expit(2.0 * (np.expm1(x) - z_i) / self.smoothing)

    def _tabulate(self, z_i, step):
        """Grid of G_i on [0, ln(1 + z_i + 40 s)], integrated panel by panel with 8-point Gauss-Legendre.

        Generated with Claude Opus 5.5.
        """
        x_hi = np.log1p(z_i + 40 * self.smoothing)
        n = int(np.ceil(x_hi / step))
        grid = np.linspace(0.0, x_hi, n + 1)
        h = grid[1] - grid[0]
        nodes = grid[:-1, None] + 0.5 * h * (_GL_X[None, :] + 1)
        panels = 0.5 * h * (self._dG(nodes, z_i) @ _GL_W)
        values = np.concatenate([[0.0], np.cumsum(panels)])
        return grid, values, self._dG(grid, z_i), z_i

    def _G(self, table, x):
        """G_i at arbitrary x ≥ 0 from its table (cubic Hermite inside, linear with slope 2 above).

        Generated with Claude Opus 5.5.
        """
        grid, values, slopes, _ = table
        h = grid[1] - grid[0]
        x = np.asarray(x, dtype=float)
        k = np.clip(((x - grid[0]) / h).astype(np.int64), 0, grid.size - 2)
        t = (x - grid[k]) / h
        t2, t3 = t * t, t * t * t
        inside = ((2 * t3 - 3 * t2 + 1) * values[k] + (t3 - 2 * t2 + t) * h * slopes[k]
                  + (-2 * t3 + 3 * t2) * values[k + 1] + (t3 - t2) * h * slopes[k + 1])
        above = values[-1] + 2.0 * (x - grid[-1])
        return np.where(x >= grid[-1], above, inside)

    def w_terms(self, z):
        """(c, B) with w(z) = c + B @ θw.

        Generated with Claude Opus 5.5.
        """
        z = np.asarray(z, dtype=float)
        n = self.edges.size
        steps = np.stack([0.5 * (1 + np.tanh((z - e) / self.smoothing)) for e in self.edges], axis=-1)
        B = np.zeros(z.shape + (n + 1,))
        B[..., 0] = 1.0
        B[..., :n] -= steps
        B[..., 1:] += steps
        return np.zeros(z.shape), B

    def lnf_terms(self, x):
        """(c, B) with ln f_DE(x) = c + B @ θw.

        Generated with Claude Opus 5.5.
        """
        x = np.asarray(x, dtype=float)
        n = self.edges.size
        G = np.stack([self._G(table, x) for table in self._tables], axis=-1)
        B = np.zeros(x.shape + (n + 1,))
        B[..., 0] = 3 * x
        B[..., :n] -= 1.5 * G
        B[..., 1:] += 1.5 * G
        return 3 * x, B


def model_from_tag(tag, edges=(0.7, 1.4, 2.1), smoothing=0.02):
    """Inverse of `model.tag`; `edges` and `smoothing` are used only for "binned".

    Generated with Claude Opus 5.5.
    """
    if tag == "lcdm":
        return LCDMModel()
    if tag == "cpl":
        return CPLModel()
    if tag == "binned":
        return BinnedWModel(edges, smoothing)
    raise ValueError(f"unknown model tag {tag!r}; use lcdm, cpl or binned")


def cosmo_param_names(likelihood_names):
    """Free cosmological and nuisance parameters, in the order of `build_parameter_spec` in
    src/minimize_GPTv2.jl: Om0, h, then M_nuisance (Pantheon+SH0ES), sigma8_0 (RSD), Ob0 (Planck or DESI).

    Generated with Claude Opus 5.5.
    """
    names = ["Om0", "h"]
    if "PantheonPlus_SH0ES" in likelihood_names:
        names.append("M_nuisance")
    if "Growth_RSD" in likelihood_names:
        names.append("sigma8_0")
    if any(n.startswith("Planck") or n == "DESI_BAO" for n in likelihood_names):
        names.append("Ob0")
    return names


def param_names(model, likelihood_names):
    """All sampled parameter names: the cosmological ones followed by `model.w_names`.

    Generated with Claude Opus 5.5.
    """
    return cosmo_param_names(likelihood_names) + list(model.w_names)


def param_bounds(model, likelihood_names):
    """Flat prior bounds of every sampled parameter, in the order of `param_names`.

    Generated with Claude Opus 5.5.
    """
    return [COSMO_PRIORS[n] for n in cosmo_param_names(likelihood_names)] + list(model.w_bounds)
