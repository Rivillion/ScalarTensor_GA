#!/bin/bash
# GA grid: 4 seeds x 4 crossover rates x 4 mutation rates = 64 tasks. See README.md.
# ADAPT: partition, QoS, wall time, cores and memory. If a list below changes length, update --array.
#SBATCH --job-name=ga-grid
#SBATCH --partition=general
#SBATCH --qos=regular
#SBATCH --array=0-63
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=5G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --output=logs/ga_grid_%A.%a.out
#SBATCH --error=logs/ga_grid_%A.%a.err
# ADAPT: for mails, remove one '#' from the next two lines and set your address.
##SBATCH --mail-user=your.name@example.org
##SBATCH --mail-type=END,FAIL
#SBATCH --signal=B:USR1@180

set -euo pipefail

# Output goes to runs/<CAMPAIGN>/. Change the name to start a new set of runs.
CAMPAIGN="grid_main"
# ADAPT: where Julia keeps its packages.
DEPOT_DIR="/scratch/${USER}/Julia_depot"

SEEDS=(104729 130363 181081 206369)
CROSSOVER_RATES=(0.20 0.40 0.60 0.80)
MUTATION_RATES=(0.20 0.40 0.60 0.80)

task="${SLURM_ARRAY_TASK_ID}"
ns="${#SEEDS[@]}"
nc="${#CROSSOVER_RATES[@]}"
seed="${SEEDS[$((task % ns))]}"
crossover="${CROSSOVER_RATES[$(((task / ns) % nc))]}"
mutation="${MUTATION_RATES[$((task / (ns * nc)))]}"

# ADAPT: how Julia 1.12 is loaded on your cluster.
module purge
module load Julia/1.12.6

export SLURM_EXPORT_ENV=ALL
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export JULIA_DEPOT_PATH="${DEPOT_DIR}"

cd "${SLURM_SUBMIT_DIR}"
out="${PWD}/runs/${CAMPAIGN}/task_${task}"
lock="${out}/LOCK"
stop="${out}/STOP"
mkdir -p "${DEPOT_DIR}" "${out}"
echo "Task ${task}: seed ${seed}, crossover ${crossover}, mutation ${mutation}, output ${out}"

# Skip a completed task, and a task that another job is still running (LOCK holds that job's id).
if [[ -e "${out}/COMPLETED" ]]; then
    echo "Already completed."
    exit 0
fi
owner="$(cat "${lock}" 2>/dev/null || true)"
if [[ -n "${owner}" && "${owner}" != "${SLURM_JOB_ID}" ]] &&
    [[ -n "$(squeue -h -j "${owner}" -t RUNNING,SUSPENDED,COMPLETING -o %i 2>/dev/null || true)" ]]; then
    echo "Still running in job ${owner}."
    exit 0
fi
rm -f "${lock}" "${stop}"
if ! (set -o noclobber; echo "${SLURM_JOB_ID}" > "${lock}") 2>/dev/null; then
    echo "Just taken by another job."
    exit 0
fi
trap 'rm -f "${lock}" "${stop}"' EXIT

# 180 s before the wall-time limit SLURM sends SIGUSR1: create the stop file, and Julia saves the run and exits (75).
export GA_STOP_FILE="${stop}"
trap 'touch "${stop}"' USR1

# --auto resumes from the last valid checkpoint, or starts a new run.
srun --export=ALL julia --gcthreads=1 --check-bounds=yes --project=. ga_run.jl --auto \
    --seed "${seed}" --crossover-rate "${crossover}" --mutation-rate "${mutation}" --output-root "${out}" &
pid=$!

# The trap interrupts `wait` (status above 128): keep waiting until Julia has saved and exited.
while :; do
    status=0
    wait "${pid}" || status=$?
    if ((status > 128)) && kill -0 "${pid}" 2>/dev/null; then continue; fi
    break
done

if ((status == 75)); then
    echo "Stopped and saved. Submit again to continue."
    exit 0
fi
exit "${status}"
