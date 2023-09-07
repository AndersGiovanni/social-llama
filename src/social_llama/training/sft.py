"""Fine-tuning script for SFT."""

import os
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import torch
from peft import AutoPeftModelForCausalLM
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import HfArgumentParser
from transformers import TrainingArguments
from trl import SFTTrainer

from social_llama.data_processing.social_dimensions import SocialDimensions


@dataclass
class ScriptArguments:
    """Script arguments."""

    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"}
    )
    log_with: Optional[str] = field(
        default="wandb", metadata={"help": "use 'wandb' to log with wandb"}
    )

    dataset_name: Optional[str] = field(
        default="social_dimensions", metadata={"help": "the dataset name"}
    )
    subset: Optional[str] = field(
        default="data/finetune", metadata={"help": "the subset to use"}
    )
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(
        default=4000, metadata={"help": "the size of the validation set"}
    )
    shuffle_buffer: Optional[int] = field(
        default=5000, metadata={"help": "the shuffle buffer size"}
    )
    seq_length: Optional[int] = field(
        default=1024, metadata={"help": "the sequence length"}
    )
    num_workers: Optional[int] = field(
        default=4, metadata={"help": "the number of workers"}
    )

    max_steps: Optional[int] = field(
        default=500, metadata={"help": "the maximum number of sgd steps"}
    )
    logging_steps: Optional[int] = field(
        default=10, metadata={"help": "the logging frequency"}
    )
    save_steps: Optional[int] = field(
        default=10, metadata={"help": "the saving frequency"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "the per device train batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the per device eval batch size"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=2, metadata={"help": "the gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )
    group_by_length: Optional[bool] = field(
        default=False, metadata={"help": "whether to group by length"}
    )
    packing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use packing for SFTTrainer"}
    )

    lora_alpha: Optional[float] = field(
        default=16, metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    learning_rate: Optional[float] = field(
        default=1e-4, metadata={"help": "the learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    num_warmup_steps: Optional[int] = field(
        default=100, metadata={"help": "the number of warmup steps"}
    )
    weight_decay: Optional[float] = field(
        default=0.05, metadata={"help": "the weight decay"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )

    output_dir: Optional[str] = field(
        default="./results", metadata={"help": "the output directory"}
    )
    log_freq: Optional[int] = field(
        default=1, metadata={"help": "the logging frequency"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    use_auth_token=True,
)
base_model.config.use_cache = False

peft_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer = AutoTokenizer.from_pretrained(
    script_args.model_name, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training


training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    group_by_length=script_args.group_by_length,
    lr_scheduler_type=script_args.lr_scheduler_type,
    warmup_steps=script_args.num_warmup_steps,
    optim=script_args.optimizer_type,
    bf16=True,
    remove_unused_columns=False,
    run_name="sft_llama2",
)

if script_args.dataset_name == "social_dimensions":
    dataset = SocialDimensions(task="zero-shot")

dataset.get_data()
train_dataset, eval_dataset = dataset.preprocess_sft(tokenizer=tokenizer)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(script_args.output_dir)

output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir, device_map="auto", torch_dtype=torch.bfloat16
)
model = model.merge_and_unload()

output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
