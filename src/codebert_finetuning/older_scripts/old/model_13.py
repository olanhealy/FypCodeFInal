import os
import json
import torch
import logging
import sys
import numpy as np
import random
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import torch.nn as nn

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "model_13.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(" Starting CodeBERT training on SSTUBS4J dataset...")

# ---------------------------
# SET DEVICE CONFIGURATION
# ---------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# DEFINE DATASET PATHS
# ---------------------------
data_dir = os.path.expanduser("~/fyp/BugLocaliser/Data/old/")
train_file = os.path.join(data_dir, "sstubsLarge-train.json")
val_file = os.path.join(data_dir, "sstubsLarge-val.json")

# ---------------------------
# LOAD TOKENISER
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# ---------------------------
# DATA PROCESSING FUNCTION
# ---------------------------
def augment_text(text):
    """Apply minor noise to text for generalisation."""
    if random.random() < 0.3:
        text = text.replace("Buggy Code:", "Code Change:")
    if random.random() < 0.2:
        text = text.replace("\n", " ")
    return text

def load_and_tokenize(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        formatted_data.append({
            "text": augment_text(f"Buggy Code: {item['sourceBeforeFix']} \nFixed Code: {item['sourceAfterFix']}"),
            "label": 1  # Label 1 means buggy
        })

    dataset = Dataset.from_list(formatted_data)
    tokenized = tokenizer(dataset["text"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = dataset["label"]
    tokenized_dataset = Dataset.from_dict(tokenized)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

logging.info(" Loading and tokenising datasets with injected noise...")
train_data = load_and_tokenize(train_file)
val_data = load_and_tokenize(val_file)

# ---------------------------
# DEFINE CUSTOM MODEL CLASS
# ---------------------------
class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.6) 
        for param in self.roberta.encoder.layer[:8].parameters():
            param.requires_grad = False  # Freeze first 8 layers

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = self.dropout(outputs.logits) 
        logits = self.dropout(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, logits)  
        return logits  

# ---------------------------
# LOAD MODEL
# ---------------------------
model = CustomRobertaForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2
)
model.to(device)
logging.info(" Model loaded with frozen layers & strong regularisation.")

# ---------------------------
# TRAINING ARGUMENTS
# ---------------------------
training_args = TrainingArguments(
    output_dir=os.path.expanduser("~/fyp/BugLocaliser/fine_tuned_model"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-6,  
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.2,
    warmup_steps=1000,  
    lr_scheduler_type="cosine",
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=True,
    logging_steps=100,
    eval_steps=500,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=os.path.expanduser("~/fyp/BugLocaliser/logs"),
)

# ---------------------------
# TRAINER INITIALISATION
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  
)

# ---------------------------
# TRAIN MODEL
# ---------------------------
logging.info(" Starting training...")
trainer.train()
logging.info(" Training complete. Best model saved.")

