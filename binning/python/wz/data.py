"""Data loaders (same files, masks and covariances as src/likelihoods.jl of the GA). Paths point at the
repository's data/ directory.

Covariances are factorised once: each correlated likelihood stores the inverse Cholesky factor L⁻¹ (lower
triangular, C = L Lᵀ), so χ² = |L⁻¹ r|².
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.integrate
import scipy.linalg

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_LIKELIHOODS = ("PantheonPlus_SH0ES", "DESI_BAO", "Planck_Zhai", "Growth_RSD")
SUPPORTED_LIKELIHOODS = ("PantheonPlus_SH0ES", "PantheonPlus_Marg", "DESI_BAO", "Planck_Zhai", "Planck_Bansal",
                         "Growth_RSD")

BAO_FLAGS = {"DV_over_rs": 1, "DM_over_DH": 2, "DM_over_rs": 3, "DH_over_rs": 4}


def inverse_cholesky(cov):
    """L⁻¹ for C = L Lᵀ, retrying with 1e-8 added to the diagonal if the Cholesky factorisation fails
    (as in src/likelihoods.jl).

    Generated with Claude Opus 5.5.
    """
    cov = np.asarray(cov, dtype=float)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]))
    return scipy.linalg.solve_triangular(L, np.eye(cov.shape[0]), lower=True)


@dataclass(frozen=True)
class PantheonPlusSH0ES:
    """Pantheon+ with SH0ES Cepheid calibrators and a free absolute magnitude M.

    Generated with Claude Opus 5.5.
    """

    name: str
    z_cmb: np.ndarray          # zHD of every kept row
    zhel_plus_one: np.ndarray
    mb_obs: np.ndarray
    is_calibrator: np.ndarray  # bool
    mu_cepheid: np.ndarray
    inv_chol: np.ndarray       # Fortran-ordered lower-triangular L⁻¹


@dataclass(frozen=True)
class PantheonPlusMarg:
    """Pantheon+ without SH0ES, zHD > 0.01, with M marginalised analytically.

    Generated with Claude Opus 5.5.
    """

    name: str
    z_cmb: np.ndarray
    zhel_plus_one: np.ndarray
    mb_obs: np.ndarray
    inv_chol: np.ndarray
    inv_chol_ones: np.ndarray  # L⁻¹ 1


@dataclass(frozen=True)
class DESIBAO:
    """DESI DR2 BAO: D_V/r_d, D_M/D_H, D_M/r_d and D_H/r_d with their full covariance.

    Generated with Claude Opus 5.5.
    """

    name: str
    z: np.ndarray
    value: np.ndarray
    flag: np.ndarray           # 1: DV/rd, 2: DM/DH, 3: DM/rd, 4: DH/rd
    inv_chol: np.ndarray


@dataclass(frozen=True)
class PlanckCompressed:
    """Compressed CMB likelihood on (R, l_a, ω_b).

    Generated with Claude Opus 5.5.
    """

    name: str
    obs: np.ndarray
    inv_chol: np.ndarray


@dataclass(frozen=True)
class GrowthRSD:
    """fσ8 compilation: independent points plus the correlated WiggleZ and BOSS DR12 LRG subsets, with the
    fiducial E(z) D_C(z) of each measurement for the fiducial-cosmology correction (as George_version's
    src/likelihoods.jl).

    Generated with Claude Opus 5.5.
    """

    name: str
    z: np.ndarray
    fs8_obs: np.ndarray
    fs8_err: np.ndarray
    idx_diag: np.ndarray
    idx_wig: np.ndarray
    inv_chol_wig: np.ndarray
    idx_boss: np.ndarray
    inv_chol_boss: np.ndarray
    EDc_fid: np.ndarray


def _read_pantheon_table(path):
    """Columns of the whitespace-separated Pantheon+SH0ES table as a dict of arrays (strings stay strings).

    Generated with Claude Opus 5.5.
    """
    with open(path) as f:
        header = f.readline().split()
        rows = [line.split() for line in f if line.strip()]
    columns = {}
    for k, name in enumerate(header):
        column = [row[k] for row in rows]
        try:
            columns[name] = np.array(column, dtype=float)
        except ValueError:
            columns[name] = np.array(column)
    return columns


@lru_cache(maxsize=1)
def _pantheon_files():
    """Pantheon+SH0ES table and the full 1701 × 1701 STAT+SYS covariance, read once per process.

    Generated with Claude Opus 5.5.
    """
    table = _read_pantheon_table(DATA_DIR / "Pantheon+SH0ES.dat")
    raw = np.fromfile(DATA_DIR / "Pantheon+SH0ES_STAT+SYS.cov", sep=" ")
    n = int(raw[0])
    if raw.size != n * n + 1:
        raise ValueError(f"Pantheon+ covariance holds {raw.size - 1} values, expected {n}²")
    # Julia reshapes column-major; the matrix is symmetric, so the row-major reshape is the same matrix.
    return table, raw[1:].reshape(n, n)


def load_pantheon_sh0es():
    """Rows with IS_CALIBRATOR == 1 or zHD > 0 and the matching covariance block.

    Generated with Claude Opus 5.5.
    """
    t, cov = _pantheon_files()
    mask = (t["IS_CALIBRATOR"] == 1) | (t["zHD"] > 0.0)
    idx = np.flatnonzero(mask)
    return PantheonPlusSH0ES("PantheonPlus_SH0ES", t["zHD"][idx], 1.0 + t["zHEL"][idx], t["m_b_corr"][idx],
                             t["IS_CALIBRATOR"][idx] == 1, t["CEPH_DIST"][idx],
                             np.asfortranarray(inverse_cholesky(cov[np.ix_(idx, idx)])))


def load_pantheon_marg():
    """Rows with zHD > 0.01 and the matching covariance block.

    Generated with Claude Opus 5.5.
    """
    t, cov = _pantheon_files()
    idx = np.flatnonzero(t["zHD"] > 0.01)
    inv_chol = np.asfortranarray(inverse_cholesky(cov[np.ix_(idx, idx)]))
    return PantheonPlusMarg("PantheonPlus_Marg", t["zHD"][idx], 1.0 + t["zHEL"][idx], t["m_b_corr"][idx],
                            inv_chol, inv_chol @ np.ones(idx.size))


def load_desi_bao():
    """DESI DR2 BAO means (z, value, quantity) and covariance; not the _leandros files.

    Generated with Claude Opus 5.5.
    """
    z, value, flag = [], [], []
    with open(DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_mean.txt") as f:
        for line in f:
            line = line.split("#", 1)[0].split()
            if not line:
                continue
            z.append(float(line[0]))
            value.append(float(line[1]))
            flag.append(BAO_FLAGS[line[2]])
    cov = np.loadtxt(DATA_DIR / "desi_gaussian_bao_ALL_GCcomb_cov.txt")
    return DESIBAO("DESI_BAO", np.array(z), np.array(value), np.array(flag), inverse_cholesky(cov))


def fiducial_E_Dc(z, Om_fid):
    """Dimensionless E(z) D_C(z) of flat ΛCDM with matter density `Om_fid`, D_C = ∫_0^z dz'/E (the denominator of
    the fiducial-cosmology correction; `fiducial_E_Dc` in src/likelihoods.jl).

    Generated with Claude Opus 5.5.
    """
    E_fid = lambda x: np.sqrt(Om_fid * (1.0 + x) ** 3 + 1.0 - Om_fid)
    D_fid, _ = scipy.integrate.quad(lambda x: 1.0 / E_fid(x), 0.0, z, epsabs=0.0, epsrel=1e-10)
    return E_fid(z) * D_fid


def load_growth_rsd():
    """fσ8 table (z, fs8, err, corr_block, Omega_m, optional source), the 3 × 3 WiggleZ covariance and the fσ8
    block of the 6 × 6 SDSS DR12 LRG covariance (its 3rd and 6th entries, at z = 0.38 and 0.51).

    Generated with Claude Opus 5.5.
    """
    rows = []
    with open(DATA_DIR / "growth_data.txt") as f:
        for line in f:
            fields = line.split("#", 1)[0].split()
            if fields:
                rows.append([float(v) for v in fields[:5]])
    rows = np.array(rows)
    z, block = rows[:, 0], rows[:, 3].astype(int)
    idx_boss = np.flatnonzero(block == 2)
    if not np.array_equal(z[idx_boss], [0.38, 0.51]):
        raise ValueError(f"expected the BOSS block at z = 0.38 and 0.51, in that order; got {z[idx_boss]}")
    cov_wig = np.loadtxt(DATA_DIR / "wigglez_cov.txt", comments="#")
    cov_sdss = np.loadtxt(DATA_DIR / "sdss_DR12_LRG_FSBAO_DMDHfs8_covtot.txt", comments="#")
    if cov_sdss.shape != (6, 6):
        raise ValueError("sdss_DR12_LRG_FSBAO_DMDHfs8_covtot.txt is not a 6 × 6 matrix")
    EDc_fid = np.array([fiducial_E_Dc(zi, om) for zi, om in zip(z, rows[:, 4])])
    return GrowthRSD("Growth_RSD", z, rows[:, 1], rows[:, 2], np.flatnonzero(block == 0),
                     np.flatnonzero(block == 1), inverse_cholesky(cov_wig), idx_boss,
                     inverse_cholesky(cov_sdss[np.ix_([2, 5], [2, 5])]), EDc_fid)


def planck_zhai():
    """Planck 2018 compressed likelihood of Zhai et al. (the GA's :Planck_Zhai).

    Generated with Claude Opus 5.5.
    """
    cov = 1.0e-8 * np.array([[1598.9554, 17112.007, -36.311179],
                             [17112.007, 811208.45, -494.79813],
                             [-36.311179, -494.79813, 2.1242182]])
    return PlanckCompressed("Planck_Zhai", np.array([1.74963, 301.80845, 0.02237]), inverse_cholesky(cov))


def planck_bansal():
    """Planck 2018 + ACT DR6 compressed likelihood of Bansal et al. (the GA's :Planck_Bansal).

    Generated with Claude Opus 5.5.
    """
    cov = 1.0e-8 * np.array([[1559.83, -1325.41, -36.45],
                             [-1325.41, 714691.80, 269.77],
                             [-36.45, 269.77, 2.10]])
    return PlanckCompressed("Planck_Bansal", np.array([1.7504, 301.77, 0.022371]), inverse_cholesky(cov))


_LOADERS = {
    "PantheonPlus_SH0ES": load_pantheon_sh0es,
    "PantheonPlus_Marg": load_pantheon_marg,
    "DESI_BAO": load_desi_bao,
    "Planck_Zhai": planck_zhai,
    "Planck_Bansal": planck_bansal,
    "Growth_RSD": load_growth_rsd,
}


@lru_cache(maxsize=None)
def load_likelihoods(names=DEFAULT_LIKELIHOODS):
    """Load the named data sets (a tuple of names, so the result is cached per process).

    Generated with Claude Opus 5.5.
    """
    names = tuple(names)
    if len(set(names)) != len(names):
        raise ValueError("likelihood names must be unique")
    unknown = [n for n in names if n not in _LOADERS]
    if unknown:
        raise ValueError(f"unsupported likelihoods {unknown}; supported: {', '.join(SUPPORTED_LIKELIHOODS)}")
    return tuple(_LOADERS[n]() for n in names)
