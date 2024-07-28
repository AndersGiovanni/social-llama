#!/bin/bash

#SBATCH --job-name=full_pipeline_single    # Job name
#SBATCH --output=run_outputs/full_pipeline_single.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=12       # Schedule one core
#SBATCH --time=2-24:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --partition=brown,red    # Run on either the Red or Brown queue
#SBATCH --mail-type=BEGIN,FAIL,END    # Send an email when the job finishes or fails
#SBATCH --account=researchers

#!/bin/bash

# Array of files
files=(
  "hahackathon#is_humor"
  "sarc"
  "contextual-abuse#IdentityDirectedAbuse"
  "contextual-abuse#PersonDirectedAbuse"
  "tweet_irony"
  "questionintimacy"
  "tweet_emotion"
  "hateoffensive"
  "implicit-hate#explicit_hate"
  "implicit-hate#implicit_hate"
  "crowdflower"
  "dailydialog"
  "hasbiasedimplication"
  "implicit-hate#stereotypical_hate"
  "intentyn"
  "tweet_offensive"
  "empathy#distress_bin"
  "complaints"
  "hayati_politeness"
  "stanfordpoliteness"
  "hypo-l"
  "rumor#rumor_bool"
  "two-to-lie#receiver_truth"
)

# model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_name="meta-llama/Llama-2-7b-chat-hf"
# model_name_short="Meta-Llama-3-8B-Instruct"
model_name_short="Llama-2-7b-chat-hf"
# Iterate over the files and corresponding scripts
for file in "${files[@]}"; do

  echo "Processing $file"
  python -m src.social_llama.training.sft --individual_task=$file --model_name=$model_name
  python -m src.social_llama.training.dpo --individual_task=$file --base_model=$model_name --model_name_or_path="sft/${model_name_short}_socket_${file}/final_checkpoint"
  python -m src.social_llama.training.merge_peft_adapter --base_model=$model_name --adapter_model_name="./dpo/${model_name_short}_socket_${file}/final_checkpoint" --output_name="models/individual/${model_name_short}_${file}"
  python -m src.social_llama.evaluation.evaluator "models/individual/${model_name_short}_${file}" $file
done
