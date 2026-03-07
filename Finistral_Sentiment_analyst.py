# =========================
# Finistral-7B Financial Sentiment Analyst: Fine-Tuning Script
# =========================

# Install required libraries 
# !pip install torch transformers peft datasets accelerate deepspeed huggingface_hub

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model     # type: ignore
from huggingface_hub import HfApi, login

# 1. Hugging Face Authentication 
login(token="YOUR_HF_TOKEN_HERE")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32    = False
set_seed(42)

# 2. Loading and preprocessing the FinGPT Sentiment dataset 
dataset = load_dataset("FinGPT/fingpt-sentiment-train")
split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
train_data = split_dataset["train"]
val_data   = split_dataset["test"]

def format_example(example):
    return {
        "instruction": "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}",
        "input":       example["input"],
        "output":      example["output"]
    }

train_data = train_data.map(format_example, remove_columns=dataset["train"].column_names)
val_data   = val_data.map(format_example,   remove_columns=dataset["train"].column_names)





# 3. Loading tokenizer & model
base_model = "mistralai/Mistral-7B-v0.1"
tokenizer  = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

tokenizer.add_special_tokens({
    
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>"
})

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    
    base_model,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
)


# 4. Applying LoRA adapter and Peft
model.gradient_checkpointing_enable()

lora_cfg = LoraConfig(
    
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_cfg)
model.enable_xformers_memory_efficient_attention()





# 5. Tokenization and mask prompt for train_on_inputs=False 
def tokenize_and_mask(ex):
    instr   = ex["instruction"]
    inp     = ex["input"]
    out     = ex["output"]
    prompt  = f"[INST]{inp}\n{instr} [/INST]"
    full    = prompt + " " + out
    tok_full   = tokenizer(full, truncation=True, max_length=4096, padding=False)
    tok_prompt = tokenizer(prompt, truncation=True, max_length=4096, padding=False)
    prompt_len = len(tok_prompt["input_ids"])
    labels     = [-100] * prompt_len + tok_full["input_ids"][prompt_len:]
    tok_full["labels"] = labels + [-100] * (len(tok_full["input_ids"]) - len(labels))
    return tok_full

train_ds = train_data.map(tokenize_and_mask, remove_columns=train_data.column_names)
val_ds   = val_data.map(tokenize_and_mask,   remove_columns=val_data.column_names)



# 6. Data collator 
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# 7. Setting Training arguments 
training_args = TrainingArguments(
    
    output_dir="./lora-out",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    num_train_epochs=4,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    weight_decay=0.0,
    bf16=True,
    fp16=False,
    logging_steps=1,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="epoch",
)



# 8. Trainer setup 
trainer = Trainer(
    
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
)



# 9. Training the model
trainer.train()



# 9. Saving & pushing to HuggingFace
trainer.save_model("./lora-out")
tokenizer.save_pretrained("./lora-out")

hf_api = HfApi()
repo_id = "Ayansk11/Finistral-7B_lora"
hf_api.create_repo(repo_id=repo_id, exist_ok=True)
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print("Pushed to:", repo_id)
