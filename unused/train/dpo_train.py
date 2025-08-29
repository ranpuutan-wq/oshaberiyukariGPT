from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch

MODEL = "Qwen/Qwen2-7B-Instruct"

pref = load_dataset("json", data_files="train/sample_dpo.jsonl")["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

args = DPOConfig(
    output_dir="./out/dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-6,
    num_train_epochs=1,
    max_prompt_length=1024,
    max_length=1792
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,  # 省略時は内部でコピー
    args=args,
    beta=0.1,
    train_dataset=pref,
    tokenizer=tokenizer,
    prompt_field="prompt",
    chosen_field="chosen",
    rejected_field="rejected",
)

trainer.train()
trainer.save_model()
