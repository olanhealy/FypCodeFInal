import os
import json
import torch
import logging
import sys
import numpy as np
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
import torch.nn as nn
import random

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "no_commit_messages_training_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(" Starting training WITHOUT commit messages...")

# ---------------------------
# SET PATHS & ENVIRONMENT
# ---------------------------
combined_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "combined"))
train_file = os.path.join(combined_data_dir, "cleaned-train.json")
val_file = os.path.join(combined_data_dir, "cleaned-val.json")
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fine_tuned_model_no_commits"))
os.makedirs(output_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD TOKENIZER
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# ---------------------------
# DEFINE MODEL WITH STRONGER GENERALISATION
# ---------------------------
class FixedRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.5)  # Stronger regularisation
       
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = self.dropout(outputs.logits)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs[1:]
        return outputs

# ---------------------------
# FREEZE PART OF CODEBERT
# ---------------------------
model = FixedRobertaForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2
)
for param in model.roberta.encoder.layer[:6].parameters():
    param.requires_grad = False  # Freeze first 6 layers
model.to(device)
logging.info(" Model loaded without commit messages.")

# ---------------------------
# LOAD AND FILTER DATA (REMOVE COMMIT MESSAGES)
# ---------------------------
def remove_commit_messages(text):
    """Remove commit messages from dataset samples."""
    lines = text.split("\n")
    filtered_lines = [line for line in lines if not line.startswith("Commit Summary:") and not line.startswith("Parent Commit:")]
    return "\n".join(filtered_lines)

def load_and_tokenize(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    for item in data:
        item["text"] = remove_commit_messages(item["text"])  # REMOVE COMMIT MESSAGES

    dataset = Dataset.from_list(data)
    tokenized = tokenizer(dataset["text"], truncation=True, padding=True, max_length=512)
    tokenized["labels"] = dataset["label"]
    tokenized_dataset = Dataset.from_dict(tokenized)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

logging.info(" Loading and tokenising training dataset...")
combined_train = load_and_tokenize(train_file)
logging.info(" Loading and tokenising validation dataset...")
combined_val = load_and_tokenize(val_file)

# ---------------------------
# TRAINING ARGUMENTS
# ---------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.1,  
    load_best_model_at_end=True,
    fp16=True,
    logging_steps=200,
    eval_steps=500,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_train,
    eval_dataset=combined_val,
    tokenizer=tokenizer,
)

# ---------------------------
# START TRAINING
# ---------------------------
logging.info(" Starting model training WITHOUT commit messages...")
trainer.train()
logging.info(" Training complete.")

