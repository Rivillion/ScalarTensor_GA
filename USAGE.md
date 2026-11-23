# Running the code

## Data and license

- The observational data in `data/` are third-party; `data/README.md` lists their sources and the papers to cite.
- The plotting notebook can use an optional font, `fonts/NewTXMI-padded.otf`, which is not included. Without it the
  Greek letters fall back to the default math font.
- The code is released under the MIT license (`LICENSE`). The license does not cover the data in `data/`.

## Reproducing the figures

The results of the paper are included, so the figures can be made without running the GA again:

- `runs/grid_main/task_<i>/<run>/`: the 64 completed GA runs (`result.h5`, `config.toml`, `metadata.toml`), plus
  `runs/grid_main/game_polished.h5`, a cache the notebook writes and reuses.
- `binning/chains/{lcdm,cpl,binned}_*.h5`: summaries of the Nautilus chains for ΛCDM, CPL and binned w(z), written
  by `binning/python/run_nautilus.py`. The raw sampler state is not included.

To make the figures, install the plotting environment once, then run `plotting_updated.ipynb` (Jupyter with the
Julia kernel, from this folder). The figures are written to `figures/`.

```bash
julia --project=plotting_env -e 'using Pkg; Pkg.instantiate()'
```

To sample the chains again (Python 3.12 and [uv](https://docs.astral.sh/uv/)):

```bash
cd binning/python
uv run python run_nautilus.py --model cpl
```

`--model` also takes `lcdm` and `binned`.

## Running the GA

The runs of the paper were made on a SLURM cluster with `ga_grid.slurm`, a 64-task array (4 seeds × 4 crossover
rates × 4 mutation rates), each task taking about a day on 32 cores. Install the packages once, then see the options
of the entry point:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. ga_run.jl --help
```

The GA settings are in `ga_run.jl`, and the grid values in `ga_grid.slurm`. `AGENTS.md` describes how the code is
organized. Setting up the runs depends on the cluster, so if you want to run the GA yourself, feel free to contact us
(see the author list in `README.md`).
