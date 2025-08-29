from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch

MODEL = "Qwen/Qwen2-7B-Instruct"  # 例：自由に変更

train = load_dataset("json", data_files="train/sample_sft.jsonl")["train"]

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, peft_cfg)

sft_cfg = SFTConfig(
    output_dir="./out/sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    max_seq_length=2048
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    tokenizer=tokenizer,
    args=sft_cfg,
    formatting_func=lambda ex: [
        "\n".join([f"{m['role'].upper()}: {m['content']}" for m in ex["messages"]])
    ]
)

trainer.train()
trainer.save_model()
