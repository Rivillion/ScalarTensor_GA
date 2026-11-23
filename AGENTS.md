# AGENTS.md

Guidance for AI coding tools working in this folder. `USAGE.md` has the user instructions (reproducing the figures, running the GA); `README.md` presents the paper.

## What this is

A Julia genetic algorithm (GA) that searches for scalar-tensor cosmologies. Each individual is a pair of symbolic
expression trees: the non-minimal coupling `F(Phi)` and the potential `U(Phi)`. The numeric constants of each candidate
are fitted jointly with the cosmological parameters against observational likelihoods, and candidates are ranked by
chi². It runs on a SLURM cluster with Julia 1.12, as a 64-task array (4 seeds × 4 crossover rates × 4 mutation rates).

## Files

| File | Role |
|---|---|
| `ga_grid.slurm` | The only SLURM file. Maps the array task to seed and rates, then calls `ga_run.jl --auto` |
| `ga_run.jl` | Configured entry point: GA settings (`configured_run_config`), initial seeds, command line, `auto_run` |
| `ga_status.sh` | Prints the state of every task; changes nothing |
| `src/run_ga.jl` | GA loop (`run_ga`, `evolve_generation!`), `GASettings`, `DATA_PATHS`, `load_likelihoods` |
| `src/GA.jl` | Trees, crossover and mutation, simplification, penalty components |
| `src/minimize.jl` | Fit of one candidate (`optimize_individual!`), parameter bounds |
| `src/evolution.jl`, `src/background.jl` | Physics: field evolution and the `Background` observables |
| `src/likelihoods.jl` | Likelihoods and the plain-text data readers |
| `src/storage.jl` | Run folders, checkpoints, `result.h5`, resume, analysis API |

## Commands

Run from this folder. Loading the code takes about 30 s.

```sh
julia --project=. ga_run.jl --help
julia --project=. ga_run.jl --dry-run --seed 1      # validate and print the configuration, no run
sbatch ga_grid.slurm                                # start, or continue everything that is unfinished
./ga_status.sh
```

There is no test suite. For a quick local check, call `auto_run` or `run_ga` with a small `Npops` and `Ngens`; the
configured values (1000 × 2000) take about a day on 32 cores.

## How it fits together

- **Not a package.** `ga_run.jl` includes `src/run_ga.jl`, which includes `GA.jl`, then `minimize.jl` (which includes
  the evolution, background and likelihood files), then `storage.jl`. Everything lands in `Main`, and the include
  order matters.
- **One generation** (`evolve_generation!`): tournament selection, breeding (`generate_children`), simplification and
  fit of every child (`evaluate_individuals!`), then replacement of the losers. The best accepted candidate is never
  replaced.
- **Fit** (`minimize.jl`): the parameter vector is `[cosmological parameters..., F constants..., U constants...]`,
  minimized with Nelder–Mead under box bounds (`_PARAMETER_BOUNDS`) and a wall-clock `time_limit` per fit. Failures
  are returned as status symbols, not thrown.
- **Physics**: the background is integrated in `N = ln a` from −21 to 0. `find_U0!` shoots the constant `U0` so that
  `E(0) = 1`. Units: `H0 = 100h` km/s/Mpc, `c` in km/s.
- **Storage**: each run writes `<output-root>/<UTCstamp>_<uuid>/` with `config.toml`, `metadata.toml`, `events.jsonl`,
  `progress.h5`, `result.h5` and `checkpoints/`. Read results with `load_run_result(path)`, `best_candidate(result)`
  and `reconstruct_background(candidate, result.config)`.

## Stopping and resuming

- `ga_grid.slurm` gives every task the folder `runs/<CAMPAIGN>/task_<i>/`. `auto_run` (`ga_run.jl`) skips a completed
  task, resumes from the newest checkpoint that passes its checksum, or starts a new run. Run folders without a usable
  checkpoint are moved to `abandoned/`.
- **Stops use a file, not a signal.** Before the wall-time limit the SLURM script creates the file named by
  `GA_STOP_FILE`. `run_ga` checks it after each generation, saves the run as `interrupted` and exits with code 75. Do
  not replace this with SIGINT handling: interrupts crash Julia 1.12 when `Base.exit_on_sigint(false)` is set.
- `LOCK` holds the id of the job running a task, and `COMPLETED` marks a finished task.
- A run resumes only with the settings it was started with (`settings_differences`). Changed settings need a new
  `CAMPAIGN` name.

## Gotchas

- **Operators are hard-coded by index**: `+`, `-`, `*` and `exp` in `GA.jl`. Several functions switch on the operator
  index, so adding or reordering operators needs coordinated edits. Seed strings accept only `Phi`, `+`, `-`, `*` and
  `exp`; write a square as `Phi * Phi`.
- **Only non-integer constants are fitted.** Integer-valued constants (`1.0`, `0.0`, ...) are structural, which is why
  the seeds use values such as `1.01`.
- **The complexity penalty is switched off**: `penalty_components` in `GA.jl` returns `total = 0.0`, so
  `description_length` equals chi².
- **Threading**: the fits run in `Threads.@threads :static`, and each thread has its own likelihood copies and cache,
  indexed by `Threads.threadid()`. Keep the `:static` schedule. Likelihood data is shared between threads and must
  stay read-only.
- **A `Background` aliases the solution of its cache.** Building another one on the same cache invalidates it; use
  `reconstruct_background` to hold several at once.
- **Runs are not bit-reproducible**, even from the same seed or checkpoint, because the fit time limit is wall-clock.
- Functions written by an AI carry a `Generated with <model>.` line in their docstring. Keep this convention.
