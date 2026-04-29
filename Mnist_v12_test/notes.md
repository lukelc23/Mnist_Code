Uses --save model

This model performs 160 experiments on TI over 8 different parameter settings and 20 seeds

Changes to make in the next version:
* To get the arrayid_taskid format, change %j to %A_%a in generic.sh:

#SBATCH --output=slurm_logs/slurm-%A_%a.out
#SBATCH --error=slurm_logs/slurm-%A_%a.err