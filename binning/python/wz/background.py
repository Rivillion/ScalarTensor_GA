"""Flat GR background with radiation, matter and a smooth dark-energy fluid (GA conventions).

Every quantity is a quadrature or a linear ODE on grids that do not depend on the parameters, so the expensive
parts (node placement and the model's ln f_DE and w terms at the nodes) are computed once in `BackgroundGrid`:

- D_M(z) = (c/H0) ∫_0^z dz'/E on z ≤ z_ref (beyond every datum and bin edge): Gauss-Legendre on the intervals
  between sorted breakpoints. The breakpoints include every requested redshift, a 0.01 grid and a s/20 grid
  within ±25 s of each bin edge, so D_M at a requested redshift is a cumulative sum, without interpolation.
- D_M(z_*) adds ∫_{z_ref}^{z_*} in x = ln(1+z), where w is constant (binned) or smooth (CPL).
- r_s(z) = ∫_0^{a(z)} c_s/(a² H) da with Gauss-Legendre on geometric panels in a, down to a = 0.
- δ'' + (2 + dlnH/dN) δ' - 1.5 Ωm(a) δ = 0 is linear, so each RK4 step is a 2 × 2 propagator; all of them are
  built at once and chained with a parallel prefix product.
"""

import numpy as np

from .models import X_MAX

C_KMS = 299792.458
OMEGA_GAMMA = 2.47739207701433e-5
N_EFF = 3.044
Z_GROWTH_INI = 100.0

_GL6 = np.polynomial.legendre.leggauss(6)
_GL8 = np.polynomial.legendre.leggauss(8)
_GL24 = np.polynomial.legendre.leggauss(24)
# Geometric panels of t = a/a(z) for the sound horizon: the integrand is smooth in a, but a stiff dark energy
# (w > 1/3) makes it a fractional power of a at a → 0.
_RS_PANELS = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0])
_ZSTAR_PANELS = 4


def radiation_density_parameter(h):
    """Ωr from photons plus Neff = 3.044 massless neutrinos.

    Generated with Claude Opus 5.5.
    """
    return (1.0 + 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0) * N_EFF) * OMEGA_GAMMA / h**2


def z_star_fit(omega_m, omega_b):
    """Photon-decoupling redshift fitting formula.

    Generated with Claude Opus 5.5.
    """
    return ((391.672 * omega_m**-0.372296 + 937.422 * omega_b**-0.97966)
            / (omega_m**-0.0192951 * omega_b**-0.93681) + omega_m**-0.731631)


def z_drag_fit(omega_m, omega_b):
    """Baryon-drag redshift fitting formula.

    Generated with Claude Opus 5.5.
    """
    return (1.0 + 428.169 * omega_b**0.256459 * omega_m**0.616388 + 925.56 * omega_m**0.751615) / omega_m**0.714129


def _panel_nodes(edges, rule):
    """Gauss-Legendre nodes and weights on every interval of the sorted `edges`; also returns the interval index
    of each node.

    Generated with Claude Opus 5.5.
    """
    x, w = rule
    lo, hi = edges[:-1, None], edges[1:, None]
    half = 0.5 * (hi - lo)
    nodes = lo + half * (x[None, :] + 1)
    weights = half * w[None, :]
    owner = np.repeat(np.arange(edges.size - 1), x.size)
    return nodes.ravel(), weights.ravel(), owner


def _merge_grid(points, min_gap=1e-9):
    """Sorted unique points with near-duplicates (closer than `min_gap`) removed.

    Generated with Claude Opus 5.5.
    """
    points = np.unique(np.asarray(points, dtype=float))
    keep = np.concatenate([[True], np.diff(points) > min_gap])
    return points[keep]


def _nearest_index(grid, values):
    """Index of the grid point nearest to each value. `_merge_grid` may have kept a neighbour within 1e-9 of a
    requested point instead of the point itself, so `searchsorted` alone can land one point too far.

    Generated with Claude Opus 5.5.
    """
    right = np.clip(np.searchsorted(grid, values), 1, grid.size - 1)
    left_closer = np.abs(grid[right - 1] - values) <= np.abs(grid[right] - values)
    return np.where(left_closer, right - 1, right)


class Background:
    """Solved background for one parameter point. Built by `BackgroundGrid.solve`; holds the expansion history and
    the derived scales.

    Generated with Claude Opus 5.5.
    """

    def __init__(self, grid, theta_w, Om0, h, Ob0, Or0, Ode0, chi_bp):
        self.grid = grid
        self.model = grid.model
        self.theta_w = theta_w
        self.Om0, self.h, self.Ob0 = Om0, h, Ob0
        self.H0 = 100.0 * h
        self.omega_b = Ob0 * h * h
        self.omega_m = Om0 * h * h
        self.Or0, self.Ode0 = Or0, Ode0
        self._chi_bp = chi_bp
        self.z_star = z_star_fit(self.omega_m, self.omega_b)
        self.z_drag = z_drag_fit(self.omega_m, self.omega_b)

    def lnf(self, x):
        """ln ρ_DE/ρ_DE(0) at x = ln(1+z).

        Generated with Claude Opus 5.5.
        """
        c, B = self.model.lnf_terms(x)
        return c + B @ self.theta_w

    def E2_x(self, x, lnf=None):
        """E²(x) = Ωr e^{4x} + Ωm e^{3x} + Ω_DE f_DE.

        Generated with Claude Opus 5.5.
        """
        lnf = self.lnf(x) if lnf is None else lnf
        return self.Or0 * np.exp(4 * x) + self.Om0 * np.exp(3 * x) + self.Ode0 * np.exp(lnf)

    def E(self, z):
        """H(z)/H0.

        Generated with Claude Opus 5.5.
        """
        return np.sqrt(self.E2_x(np.log1p(np.asarray(z, dtype=float))))

    def H(self, z):
        """Hubble rate in km/s/Mpc.

        Generated with Claude Opus 5.5.
        """
        return self.H0 * self.E(z)

    def chi(self, z):
        """Dimensionless comoving distance ∫_0^z dz'/E at any z ≤ 1e5 (exact quadrature, no interpolation).

        Generated with Claude Opus 5.5.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        bp = self.grid.z_bp
        out = np.empty(z.shape)
        low = z <= bp[-1]
        if np.any(low):
            k = np.clip(np.searchsorted(bp, z[low], side="right") - 1, 0, bp.size - 1)
            out[low] = self._chi_bp[k] + self._chi_segment(bp[k], z[low], panels=1, rule=_GL8)
        if np.any(~low):
            out[~low] = self._chi_bp[-1] + self._chi_segment(np.full((~low).sum(), bp[-1]), z[~low])
        return out

    def _chi_segment(self, z_lo, z_hi, panels=_ZSTAR_PANELS, rule=_GL24):
        """∫_{z_lo}^{z_hi} dz/E, integrated in x = ln(1+z) (dz = e^x dx) on `panels` equal panels per segment.

        Generated with Claude Opus 5.5.
        """
        x_lo, x_hi = np.log1p(np.asarray(z_lo, dtype=float)), np.log1p(np.asarray(z_hi, dtype=float))
        t, w = rule
        edges = np.linspace(0.0, 1.0, panels + 1)
        total = np.zeros(np.shape(x_lo))
        for p in range(panels):
            a, b = edges[p], edges[p + 1]
            frac = a + 0.5 * (b - a) * (t + 1)                      # nodes in [0, 1]
            x = x_lo[..., None] + (x_hi - x_lo)[..., None] * frac
            f = np.exp(x) / np.sqrt(self.E2_x(x))
            total += (x_hi - x_lo) * 0.5 * (b - a) * (f @ w)
        return total

    def comoving_distance(self, z):
        """Transverse comoving distance D_M(z) in Mpc.

        Generated with Claude Opus 5.5.
        """
        return C_KMS / self.H0 * self.chi(z)

    def sound_horizon(self, z):
        """Comoving sound horizon r_s(z) = ∫_0^{a(z)} c_s/(a² H) da in Mpc, c_s = c/√(3(1 + R a)),
        R = 0.75 ω_b/Ω_γ.

        Generated with Claude Opus 5.5.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        a_max = 1.0 / (1.0 + z)
        t, w = self.grid.rs_nodes, self.grid.rs_weights
        a = a_max[:, None] * t[None, :]
        x = -np.log(a)
        # a⁴E² = Ωr + Ωm a + Ω_DE a⁴ f_DE, finite at a → 0
        a4E2 = self.Or0 + self.Om0 * a + self.Ode0 * np.exp(self.lnf(x) - 4 * x)
        R = 0.75 * self.omega_b / OMEGA_GAMMA
        f = 1.0 / np.sqrt((1.0 + R * a) * a4E2)
        return C_KMS / (np.sqrt(3.0) * self.H0) * a_max * (f @ w)

    def growth(self):
        """(δ'(N) at the grid's growth redshifts, δ(0)) with δ = δ' = a at z = 100.

        Generated with Claude Opus 5.5.
        """
        g = self.grid
        A = []
        for x, lnf_c, lnf_B, w_c, w_B in g.growth_terms:
            lnf = lnf_c + lnf_B @ self.theta_w
            w = w_c + w_B @ self.theta_w
            rho_r = self.Or0 * np.exp(4 * x)
            rho_m = self.Om0 * np.exp(3 * x)
            rho_de = self.Ode0 * np.exp(lnf)
            E2 = rho_r + rho_m + rho_de
            dlnH = (-4 * rho_r - 3 * rho_m - 3 * (1 + w) * rho_de) / (2 * E2)
            M = np.zeros(x.shape + (2, 2))
            M[:, 0, 1] = 1.0
            M[:, 1, 0] = 1.5 * rho_m / E2
            M[:, 1, 1] = -(2 + dlnH)
            A.append(M)
        A1, A2, A3 = A
        hs = g.growth_h[:, None, None]
        eye = np.eye(2)[None]
        K1 = A1
        K2 = A2 @ (eye + 0.5 * hs * K1)
        K3 = A2 @ (eye + 0.5 * hs * K2)
        K4 = A3 @ (eye + hs * K3)
        P = eye + hs / 6.0 * (K1 + 2 * K2 + 2 * K3 + K4)
        # Inclusive prefix product P[k] ← P[k] ⋯ P[0] (Hillis-Steele scan).
        d = 1
        while d < P.shape[0]:
            P = np.concatenate([P[:d], P[d:] @ P[:-d]])
            d *= 2
        a_ini = 1.0 / (1.0 + Z_GROWTH_INI)
        y0 = np.array([a_ini, a_ini])
        states = np.concatenate([y0[None], P @ y0])     # state at every grid point
        out = states[g.growth_out]
        return out[:, 1], states[-1, 0]

    def fs8(self, sigma8_0):
        """fσ8 = σ8,0 δ'(z)/δ(0) at the grid's growth redshifts.

        Generated with Claude Opus 5.5.
        """
        ddelta, delta0 = self.growth()
        return sigma8_0 * ddelta / delta0


class BackgroundGrid:
    """Parameter-independent nodes for one model and one set of requested redshifts.

    `z_dm`: redshifts where D_M is needed, `z_growth`: redshifts where fσ8 is needed.

    Generated with Claude Opus 5.5.
    """

    def __init__(self, model, z_dm=(), z_growth=(), base_dz=0.01, edge_halfwidth=25.0, edge_step=0.05,
                 growth_step=0.004, growth_edge_step=0.1, rs_rule=_GL24):
        self.model = model
        z_dm = np.asarray(z_dm, dtype=float)
        s = getattr(model, "smoothing", 0.02)
        edges = np.asarray(model.transitions, dtype=float)
        z_top = max([3.0] + ([edges[-1] + 50 * s] if edges.size else []) + ([z_dm.max() + 0.01] if z_dm.size else []))
        self.z_ref = z_top
        pieces = [np.arange(0.0, z_top, base_dz), [z_top], z_dm]
        for e in edges:
            pieces.append(np.arange(e - edge_halfwidth * s, e + edge_halfwidth * s, edge_step * s))
        bp = _merge_grid(np.concatenate([np.asarray(p, dtype=float) for p in pieces]))
        self.z_bp = bp[(bp >= 0.0) & (bp <= z_top)]
        nodes, weights, owner = _panel_nodes(self.z_bp, _GL6)
        self.dm_nodes_x = np.log1p(nodes)
        self.dm_weights = weights
        self.dm_owner = owner
        self.dm_lnf = model.lnf_terms(self.dm_nodes_x)
        self.dm_index = _nearest_index(self.z_bp, z_dm)
        if z_dm.size and not np.allclose(self.z_bp[self.dm_index], z_dm, rtol=0, atol=1e-9):
            raise RuntimeError("requested redshifts are not grid breakpoints")

        # Sound-horizon nodes in t = a/a(z).
        self.rs_nodes, self.rs_weights, _ = _panel_nodes(_RS_PANELS, rs_rule)

        # Growth: RK4 grid in N with the requested redshifts as grid points.
        z_growth = np.asarray(z_growth, dtype=float)
        n_ini = -np.log1p(Z_GROWTH_INI)
        pts = [np.arange(n_ini, 0.0, growth_step), [n_ini, 0.0], -np.log1p(z_growth)]
        for e in edges:
            width = s / (1 + e)
            pts.append(-np.log1p(e) + np.arange(-30 * width, 30 * width, growth_edge_step * width))
        grid_n = _merge_grid(np.concatenate([np.asarray(p, dtype=float) for p in pts]))
        grid_n = grid_n[(grid_n >= n_ini) & (grid_n <= 0.0)]
        self.growth_n = grid_n
        self.growth_h = np.diff(grid_n)
        self.growth_out = _nearest_index(grid_n, -np.log1p(z_growth))
        terms = []
        for n_pts in (grid_n[:-1], grid_n[:-1] + 0.5 * self.growth_h, grid_n[1:]):
            x = -n_pts
            terms.append((x, *model.lnf_terms(x), *model.w_terms(np.expm1(x))))
        self.growth_terms = terms

    def solve(self, theta_w, Om0, h, Ob0):
        """Background at one parameter point, or None if Ω_DE ≤ 0 or the expansion is not finite.

        Generated with Claude Opus 5.5.
        """
        theta_w = np.asarray(theta_w, dtype=float)
        Or0 = radiation_density_parameter(h)
        Ode0 = 1.0 - Om0 - Or0
        if not Ode0 > 0:
            return None
        x = self.dm_nodes_x
        c, B = self.dm_lnf
        E2 = Or0 * np.exp(4 * x) + Om0 * np.exp(3 * x) + Ode0 * np.exp(c + B @ theta_w)
        integrand = self.dm_weights / np.sqrt(E2)
        per_interval = np.bincount(self.dm_owner, weights=integrand, minlength=self.z_bp.size - 1)
        chi_bp = np.concatenate([[0.0], np.cumsum(per_interval)])
        if not np.all(np.isfinite(chi_bp)):
            return None
        return Background(self, theta_w, Om0, h, Ob0, Or0, Ode0, chi_bp)

    def chi_requested(self, bg):
        """∫_0^z dz/E at the `z_dm` given to the constructor.

        Generated with Claude Opus 5.5.
        """
        return bg._chi_bp[self.dm_index]


def make_background(model, theta_w, Om0, h, Ob0):
    """Convenience constructor for interactive use: a background on a generic grid, or None if unphysical.

    Generated with Claude Opus 5.5.
    """
    return BackgroundGrid(model).solve(theta_w, Om0, h, Ob0)


__all__ = ["Background", "BackgroundGrid", "make_background", "radiation_density_parameter", "z_star_fit",
           "z_drag_fit", "C_KMS", "X_MAX"]
