from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os
print("Current working directory:", os.getcwd())

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Prepare dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="cleaned_ielts_questions.txt",
    block_size=64
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./ielts_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train
trainer.train()

# Save the model
trainer.save_model("./ielts_model")
tokenizer.save_pretrained("./ielts_model")
