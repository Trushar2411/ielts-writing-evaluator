from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from tqdm import tqdm
import logging
import torch
import random
import numpy as np

# Optional: For reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 🔥 Important line
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))  # Resize embeddings

# Load dataset (CSV format with "Question" column)
dataset = load_dataset("csv", data_files="cleaned_ielts_questions.csv")

# Check available columns
print("Available columns:", dataset["train"].column_names)

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["Question"],
        truncation=True,
        padding="max_length",
        max_length=128,  # Optionally increased from 64
    )

# Tokenize with tqdm
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, desc="Tokenizing with tqdm")

# Data collator (for language modeling)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./ielts_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    report_to="none",
    fp16=torch.cuda.is_available(),  # Enable FP16 if on GPU
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

# Train
print("Training model...")
trainer.train()

# Save model and tokenizer
print("Saving model...")
trainer.save_model("./ielts_model")
tokenizer.save_pretrained("./ielts_model")

print("✅ Training complete.")
