#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Solve one PyPSA-GB scenario per array task on Barkla.
#
#   sbatch -a 0             hpc/run_scenario.slurm   # prep: first scenario alone
#   sbatch -a 1-N --dependency=afterok:<jobid> hpc/run_scenario.slurm
#
# Run the FIRST scenario on its own before launching the rest. Several
# pipeline stages write shared artefacts (FES CSVs, renewable profiles,
# interconnector tables) that are not scenario-specific; if concurrent tasks
# generate them simultaneously they will corrupt each other. Once the first
# task has completed they exist and every later task reads them read-only.
#
# --nolock is required because array tasks share the working directory and
# Snakemake would otherwise refuse to start. It is safe here only because the
# scenario targets are disjoint and the shared artefacts already exist - which
# is exactly what the prep step above guarantees.
# ---------------------------------------------------------------------------
# Specify where to run scripts from
#SBATCH -D /users/lukeben/scratch/PyPSA-GB
# Specify job name
#SBATCH -J pypsagb
# Specify where standard error and output should be sent
# %x: job name, %A: array job id, %a: task index in array
#SBATCH -o hpc/logs/%x_%A_%a.out
# Specify single scenario
#SBATCH -a 0-3
# Request the partition
#SBATCH -p nodes           # 'himem' for unclustered / full ETYS
# Request single node with 42 cores
#SBATCH -N 1 -n 42         # When using himem partition max cores = 32
# Specify max run time
#SBATCH -t 24:00:00
# Email when done
#SBATCH --mail-user=lukeBen@liverpool.ac.uk
#SBATCH --mail-type=END,FAIL


set -euo pipefail
export PYTHONUNBUFFERED=1

# The pipeline writes resources/, logs/, benchmarks/ and .snakemake/ inside
# the repo, so all of it has to sit on scratch.
CONDA_ROOT="${CONDA_ROOT:-$HOME/fastscratch/miniforge3}"
REPO="${REPO:-$HOME/scratch/PyPSA-GB}"
ENV_NAME="${ENV_NAME:-pypsa-gb}"
SCENARIO_LIST="${SCENARIO_LIST:-$REPO/hpc/scenarios.txt}"

# conda init was declined at install time, so the hook must be sourced
# explicitly.
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$REPO"
mkdir -p hpc/logs

# Strip inline comments and surrounding whitespace, then drop empty lines, so
# entries can be annotated: "HT37_counterfactual  # comparison run".
mapfile -t SCENARIOS < <(
    sed -E 's/#.*//; s/^[[:space:]]+//; s/[[:space:]]+$//; /^$/d' "$SCENARIO_LIST"
)

if [ "${#SCENARIOS[@]}" -eq 0 ]; then
    echo "ERROR: no scenarios parsed from $SCENARIO_LIST" >&2
    exit 1
fi

SCENARIO="${SCENARIOS[${SLURM_ARRAY_TASK_ID:-0}]:-}"
if [ -z "$SCENARIO" ]; then
    echo "ERROR: array index ${SLURM_ARRAY_TASK_ID:-0} out of range" \
         "(${#SCENARIOS[@]} scenarios in $SCENARIO_LIST)" >&2
    exit 1
fi

echo "============================================================"
echo "scenario   : $SCENARIO"
echo "node       : $(hostname)"
echo "cpus       : ${SLURM_NTASKS:-?}"
echo "memory     : $(free -g | awk '/^Mem:/{print $2}') GB"
echo "started    : $(date)"
echo "============================================================"

# Solver threads must match the SLURM allocation. Left at the config default
# of 4 the job would use a fortieth of the node it is holding.
export OMP_NUM_THREADS="${SLURM_NTASKS:-42}"

TARGET="resources/network/${SCENARIO}_solved.nc"

# Start timer
date_start=`date +%s`

snakemake \
    --config scenario="$SCENARIO" \
    --cores "${SLURM_NTASKS:-42}" \
    --rerun-triggers mtime \
    --nolock \
    "$TARGET"

# Generate notebooks
snakemake --config scenario="$SCENARIO" --cores 1 --rerun-triggers mtime --nolock \
    "resources/analysis/${SCENARIO}_notebook.ipynb" || \
    echo "WARNING: notebook generation failed for $SCENARIO"

# End timer
date_end=`date +%s`
seconds=$((date_end-date_start))
minutes=$((seconds/60))
seconds=$((seconds-60*minutes))
hours=$((minutes/60))
minutes=$((minutes-60*hours))

echo
echo "============================================================"
echo "finished : $(date)"
echo "total run time :  $hours Hours $minutes Minutes $seconds Seconds"
echo "============================================================"

# Post-solve diagnostics. Non-fatal: a correctness failure should be visible
# in the log without discarding a solve that took hours.
python scripts/nuclear/check_tes_run.py "$SCENARIO" || \
    echo "WARNING: check_tes_run reported failures for $SCENARIO"
