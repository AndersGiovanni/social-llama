#!/bin/bash

#SBATCH --job-name=rag    # Job name
#SBATCH --output=run_outputs/rag.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=12       # Schedule one core
#SBATCH --time=3-00:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --partition=brown,red    # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END    # Send an email when the job finishes or fails
#SBATCH --account=researchers

hostname
nvidia-smi
# python -m src.social_llama.training.rag "google/gemma-7b-it"
# python -m src.social_llama.training.rag "meta-llama/Llama-2-7b-chat-hf"
python -m src.social_llama.training.rag "AndersGiovanni/social-llama-7b-beta"
