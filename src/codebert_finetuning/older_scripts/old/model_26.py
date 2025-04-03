import os
# Force GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# ----- Configuration -----
data_files = {
    "train": "../../Data/SSTUBS4J/sstubsLarge-train-processed.json",
    "validation": "../../Data/SSTUBS4J/sstubsLarge-val-processed.json",
    "test": "../../Data/SSTUBS4J/sstubsLarge-test-processed.json"
}

# ----- Load Dataset -----
dataset = load_dataset("json", data_files=data_files)

# ----- Tokenisation -----
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(examples):
    return tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["input_text"])
tokenized_datasets.set_format("torch")

# ----- Model Setup -----
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# ----- Training Configuration -----
training_args = TrainingArguments(
    output_dir="./context_aware_bug_localisation_model",
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    eval_steps=500,
    save_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()
print("Training complete.")

results = trainer.evaluate(tokenized_datasets["test"])
print("Test set evaluation:", results)

trainer.save_model("./context_aware_bug_localisation_model")

