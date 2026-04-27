#!/bin/bash

# Single-run timing test for Mnist_v10.
# Submit with:  sbatch single_test.sh
# Output goes to ${OUTPUT_FOLDER} on the locker, plus the slurm log file.

#SBATCH --chdir=/home/lc3616/Projects/Mnist_v10/slurm-for-ml-master
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/timing-%j.out
#SBATCH --error=slurm_logs/timing-%j.err
#SBATCH --partition=abbott
#SBATCH --account=abbott
#SBATCH --time=02:00:00
#SBATCH --job-name=v10_timing

set -e

mkdir -p slurm_logs

export PYTHONPATH=/home/lc3616/Projects:$PYTHONPATH
path_to_conda="/share/apps/anaconda3-2023.07.03"

OUTPUT_FOLDER="/mnt/smb/locker/abbott-locker/Luke/Mnist/Mnist_v10/timing_test_ordering_1_exception_true_dropout_true_intermediate_layer_true"
mkdir -p "$OUTPUT_FOLDER"

echo "=========================================="
echo "v10 single-run timing test"
echo "Host:   $(hostname)"
echo "Start:  $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
echo "Output: $OUTPUT_FOLDER"
echo "=========================================="

source ${path_to_conda}/bin/activate tiexp
echo "Python: $(which python)"

# /usr/bin/time -v gives detailed timing; fall back to plain `time` if not available.
TIMER="/usr/bin/time -v"
if ! command -v /usr/bin/time >/dev/null 2>&1; then
    TIMER="time"
fi

$TIMER srun python /home/lc3616/Projects/Mnist_v10/Mnist_TI.py \
    --output-folder "$OUTPUT_FOLDER" \
    --exception true \
    --dropout true \
    --intermediate-layer true \
    --ordering-seed 1 \
    --n-items 9 \
    --epochs 100 \
    --save-model \
    --exception-pair 5 3

echo "=========================================="
echo "End:    $(date)"
echo "=========================================="

# Move slurm logs into the output folder for easy inspection later.
mv "slurm_logs/timing-${SLURM_JOB_ID}.out" "${OUTPUT_FOLDER}/" 2>/dev/null || true
mv "slurm_logs/timing-${SLURM_JOB_ID}.err" "${OUTPUT_FOLDER}/" 2>/dev/null || true
