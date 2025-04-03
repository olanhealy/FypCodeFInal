import os
import json
import random
import logging
from collections import defaultdict
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "final_train_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("Starting final training script...")

# ---------------------------
# SET PATHS & CONFIGURATION
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SSTUBS_FINAL_DIR = os.path.join(BASE_DIR, "Data", "sstubs4j_final")  # UPDATED DATASET PATH
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "fine_tuned_model")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ---------------------------
# LOAD DATASETS
# ---------------------------
def load_json(file_path):
    """Load JSON dataset."""
    with open(file_path, "r") as f:
        return json.load(f)

train_data = load_json(os.path.join(SSTUBS_FINAL_DIR, "sstubsLarge-train.json"))
val_data = load_json(os.path.join(SSTUBS_FINAL_DIR, "sstubsLarge-val.json"))
test_data = load_json(os.path.join(SSTUBS_FINAL_DIR, "sstubsLarge-test.json"))

logging.info(f"Train set size: {len(train_data)} | Val set size: {len(val_data)} | Test set size: {len(test_data)}")

# ---------------------------
# TOKENIZER & MODEL
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Convert lists to Hugging Face Dataset objects and tokenize
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

# Rename label column to "labels" as required
train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------------------
# MODEL CLASS
# ---------------------------
class MyRobertaClassifier(RobertaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.3)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = self.dropout(outputs.logits)
        if labels is not None and self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device), label_smoothing=0.1)
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs[1:]
        return outputs

class_weights = torch.tensor([1.0, 1.0]).to(device)
model = MyRobertaClassifier.from_pretrained("microsoft/codebert-base", num_labels=2, class_weights=class_weights)
model.to(device)

# ---------------------------
# TRAINING CONFIGURATION
# ---------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_OUTPUT_DIR, "results"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(MODEL_OUTPUT_DIR, "logs"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=500,
    lr_scheduler_type="cosine_with_restarts",
    load_best_model_at_end=True,
    report_to=None,
    fp16=torch.cuda.is_available()
)

# ---------------------------
# TRAINING
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

logging.info("Starting training...")
trainer.train()

# ---------------------------
# SAVE MODEL
# ---------------------------
model_dir = os.path.join(MODEL_OUTPUT_DIR, "model_final")
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
logging.info(f"Model saved to {model_dir}")

if __name__ == "__main__":
    main()

