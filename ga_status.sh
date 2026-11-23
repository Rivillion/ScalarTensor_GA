#!/bin/bash
# State of every task of a campaign, read from runs/<CAMPAIGN>/task_<i>/.
#
#   ./ga_status.sh [CAMPAIGN]
#
# Without an argument it uses the CAMPAIGN set in ga_grid.slurm. Run it on the cluster's login node: it
# asks squeue which tasks are running. It changes nothing.
set -uo pipefail
cd "$(dirname "$0")"

slurm_file="ga_grid.slurm"
campaign="${1:-$(sed -n 's/^CAMPAIGN="\(.*\)"$/\1/p' "${slurm_file}")}"
read -r first last < <(sed -n 's/^#SBATCH --array=\([0-9]*\)-\([0-9]*\).*/\1 \2/p' "${slurm_file}")
if [[ -z "${campaign}" || -z "${first:-}" || -z "${last:-}" ]]; then
    echo "Could not read CAMPAIGN or --array=FIRST-LAST from ${slurm_file}." >&2
    exit 2
fi

# Highest generation with a checkpoint among the run folders of a task; empty if there is none.
last_checkpoint() {
    cat "$1"/*/checkpoints/latest.toml 2>/dev/null | sed -n 's/^generation = \([0-9]*\)$/\1/p' | sort -n | tail -1
}

completed=0; running=0; stopped=0; unstarted=0; todo=()
echo "Campaign ${campaign}: tasks ${first}-${last}"
for ((task = first; task <= last; task++)); do
    dir="runs/${campaign}/task_${task}"
    generation="$(last_checkpoint "${dir}")"
    checkpoint="no checkpoint yet"
    if [[ -n "${generation}" ]]; then checkpoint="last checkpoint at generation ${generation}"; fi
    owner="$(cat "${dir}/LOCK" 2>/dev/null || true)"
    if [[ -e "${dir}/COMPLETED" ]]; then
        state="completed"; completed=$((completed + 1))
    elif [[ -n "${owner}" && -n "$(squeue -h -j "${owner}" -t RUNNING,SUSPENDED,COMPLETING -o %i 2>/dev/null || true)" ]]; then
        state="running in job ${owner}, ${checkpoint}"; running=$((running + 1))
    elif [[ -d "${dir}" ]]; then
        state="stopped, ${checkpoint}"; stopped=$((stopped + 1)); todo+=("${task}")
    else
        state="not started"; unstarted=$((unstarted + 1)); todo+=("${task}")
    fi
    printf 'task %3d: %s\n' "${task}" "${state}"
done

echo
echo "${completed} completed, ${running} running, ${stopped} stopped, ${unstarted} not started."
if ((${#todo[@]} > 0)); then
    list="$(IFS=,; echo "${todo[*]}")"
    echo "Tasks that are stopped or not started may also be waiting in the queue (squeue -u \$USER)."
    echo "To continue everything:        sbatch ${slurm_file}"
    echo "To submit only these tasks:    sbatch --array=${list} ${slurm_file}"
fi
