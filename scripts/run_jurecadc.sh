#!/bin/bash -x
#SBATCH --account=zam
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
source scripts/init.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Job id: $SLURM_JOB_ID"
export TOKENIZERS_PARALLELISM=false
srun python -u $*
