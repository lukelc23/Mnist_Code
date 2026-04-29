#!/bin/bash

# This is a generic running script. It can run in two configurations:
# Single job mode: pass the python arguments to this script
# Batch job mode: pass a file with first the job tag and second the commands per line

#SBATCH --chdir=/home/lc3616/Projects/Mnist_TI/Mnist_v13/slurm-for-ml-master
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/slurm-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%A_%a.err
#SBATCH --partition=ctn
#SBATCH --account=ctn
#SBATCH --time=07:00:00

set -e # fail fully on first line failure

export PYTHONPATH=/home/lc3616/Projects:$PYTHONPATH

# Customize this line to point to conda installation
path_to_conda="/share/apps/anaconda3-2023.07.03"

echo "Running on $(hostname)"

if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode

    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array

    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# Find what was passed to --output_folder
regexp="--output-folder\s+(\S+)"
if [[ $JOB_CMD =~ $regexp ]]
then
    JOB_OUTPUT=${BASH_REMATCH[1]}
else
    echo "Error: did not find a --output_folder argument"
    exit 1
fi

# Check if results exists, if so, we could remove slurm log and skip
if [ -f  "$JOB_OUTPUT/results.json" ]
then
    echo "Results already done - exiting"
    #commenting this out because it's not really necessary for me
    # rm "slurm-${JOB_ID}.out" 
    exit 0
fi

# Check if the output folder exists at all. We could remove the folder in that case.
if [ -d  "$JOB_OUTPUT" ]
then
    echo "Folder exists, but was unfinished or is ongoing (no results.json)."
    echo "Starting job as usual"
    # It might be worth removing the folder at this point:
    # echo "Removing current output before continuing"
    # rm -r "$JOB_OUTPUT"
    # Since this is a destructive action it is not on by default
fi

# Use this line if you need to create the environment first on a machine
# ./run_locked.sh ${path_to_conda}/bin/conda-env update -f environment.yml

# Activate the environment
source ${path_to_conda}/bin/activate tiexp

# Train the model
srun python $JOB_CMD

# Move the log file to the job folder
mv "slurm_logs/slurm-${JOB_ID}.out" "${JOB_OUTPUT}/"
mv "slurm_logs/slurm-${JOB_ID}.err" "${JOB_OUTPUT}/" 2>/dev/null || true
