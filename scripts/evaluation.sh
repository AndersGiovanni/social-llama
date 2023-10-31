#!/bin/bash

#SBATCH --job-name=evaluation    # Job name
#SBATCH --output=run_outputs/evaluation.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=12       # Schedule one core
#SBATCH --time=04:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=brown,red    # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END    # Send an email when the job finishes or fails
#SBATCH --account=researchers

hostname

module load poetry/1.5.1-GCCcore-12.3.0

python -m src.social_llama.evaluation.evaluator
