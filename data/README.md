# Observational data

These files are third-party public data, redistributed unchanged (except where noted) so that the results can be
reproduced. They are **not** covered by the code license in `../LICENSE`; all rights remain with their original
authors. If you use them, cite the original papers listed below.

| File | Content | Source |
|---|---|---|
| `Pantheon+SH0ES.dat` | Pantheon+ SN Ia distances, 1701 light curves | Pantheon+ data release, <https://github.com/PantheonPlusSH0ES/DataRelease> (`Pantheon+_Data/4_DISTANCES_AND_COVAR/`) |
| `Pantheon+SH0ES_STAT+SYS.cov` | Pantheon+SH0ES statistical + systematic covariance | Same as above |
| `desi_gaussian_bao_ALL_GCcomb_mean.txt` | DESI DR2 BAO measurements (galaxies, quasars and Lyα) | DESI DR2 BAO Gaussian likelihood, as distributed in <https://github.com/CobayaSampler/bao_data> (`desi_bao_dr2/`) |
| `desi_gaussian_bao_ALL_GCcomb_cov.txt` | DESI DR2 BAO covariance | Same as above |
| `sdss_DR12_LRG_FSBAO_DMDHfs8_covtot.txt` | BOSS DR12 LRG covariance of (D_M, D_H, fσ8) at z = 0.38 and 0.51 | SDSS DR16 cosmology likelihoods (Alam et al. 2021), <https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/likelihoods/BAO-plus/> |
| `wigglez_cov.txt` | WiggleZ fσ8 covariance | Blake et al. 2012, as written in Eq. (18) of Efstratiou et al. 2026 |
| `growth_data.txt` | Compilation of 20 fσ8 (RSD) measurements | Table A.1 of Ocampo et al. (Euclid) 2026; the original measurement of each row is named in its `source` column. Asymmetric errors are symmetrized (see the file header) |

The compressed Planck 2018 CMB likelihood is not a file: its distance priors (R, l_A, ω_b) and covariance are written
in `src/likelihoods.jl` (`PlanckCompressed_Zhai2018`), taken from Zhai & Wang 2019.

## References

- D. Scolnic et al., *The Pantheon+ Analysis: The Full Data Set and Light-curve Release*, ApJ 938, 113 (2022),
  [arXiv:2112.03863](https://arxiv.org/abs/2112.03863).
- D. Brout et al., *The Pantheon+ Analysis: Cosmological Constraints*, ApJ 938, 110 (2022),
  [arXiv:2202.04077](https://arxiv.org/abs/2202.04077).
- A. G. Riess et al., *A Comprehensive Measurement of the Local Value of the Hubble Constant with 1 km/s/Mpc
  Uncertainty from the Hubble Space Telescope and the SH0ES Team*, ApJL 934, L7 (2022),
  [arXiv:2112.04510](https://arxiv.org/abs/2112.04510).
- DESI Collaboration, M. Abdul Karim et al., *DESI DR2 Results II: Measurements of Baryon Acoustic Oscillations and
  Cosmological Constraints*, Phys. Rev. D 112 (2025), [arXiv:2503.14738](https://arxiv.org/abs/2503.14738).
- S. Alam et al. (BOSS), *The clustering of galaxies in the completed SDSS-III Baryon Oscillation Spectroscopic
  Survey: cosmological analysis of the DR12 galaxy sample*, MNRAS 470, 2617 (2017),
  [arXiv:1607.03155](https://arxiv.org/abs/1607.03155).
- S. Alam et al. (eBOSS), *Completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey: Cosmological
  implications from two decades of spectroscopic surveys at the Apache Point Observatory*, Phys. Rev. D 103, 083533
  (2021), [arXiv:2007.08991](https://arxiv.org/abs/2007.08991).
- C. Blake et al., *The WiggleZ Dark Energy Survey: joint measurements of the expansion and growth history at
  z < 1*, MNRAS 425, 405 (2012), [arXiv:1204.3674](https://arxiv.org/abs/1204.3674).
- D. Efstratiou, E. A. Paraskevas and L. Perivolaropoulos, *Addressing the DESI DR2 phantom-crossing anomaly and
  enhanced H0 tension with reconstructed scalar-tensor gravity*, Phys. Rev. D 113 (2026),
  [arXiv:2511.04610](https://arxiv.org/abs/2511.04610).
- Euclid Collaboration, I. Ocampo et al., *Euclid: Forecasts on ΛCDM consistency tests with growth rate data*,
  A&A 714, A57 (2026), [arXiv:2507.22780](https://arxiv.org/abs/2507.22780).
- Z. Zhai and Y. Wang, *Robust and model-independent cosmological constraints from distance measurements*, JCAP 07
  (2019) 005, [arXiv:1811.07425](https://arxiv.org/abs/1811.07425).
- Planck Collaboration, *Planck 2018 results. VI. Cosmological parameters*, A&A 641, A6 (2020),
  [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
