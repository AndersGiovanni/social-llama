#!/usr/bin/env bash
#
# A LUMI SLURM batch script for the LUMI PyTorch single GPU test example from
# https://github.com/DeiC-HPC/cotainr
#
#SBATCH --job-name=evaluator
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output="run_outputs/output_%x_%j.txt"
#SBATCH --partition=small-g
#SBATCH --time=1-24:00:00
#SBATCH --account=project_465000859

module load LUMI 
module load cotainr

touch /project/project_465000859/testfile

export HF_TOKEN=$TOKEN

export HF_HOME='/project/project_465000859/.cache/'
export HF_DATASETS_CACHE='/project/project_465000859/.cache/'
export HUGGINGFACE_HUB_CACHE='/project/project_465000859/.cache/'
export TRANSFORMERS_CACHE='/project/project_465000859/.cache/'

# Execute your Python script inside the Singularity container
srun singularity exec --env HF_TOKEN="$HF_TOKEN" $SCRATCH/lumi_pytorch_rocm.sif python src/social_llama/evaluation/evaluator.py