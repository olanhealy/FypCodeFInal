import os
import json
import torch
import logging
import sys
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch.nn as nn
import random

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "model_14_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("Starting updated training script with combined dataset...")

# ---------------------------
# SET PATHS & ENVIRONMENT
# ---------------------------
combined_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data", "combined"))
train_file = os.path.join(combined_data_dir, "cleaned-train.json")
val_file = os.path.join(combined_data_dir, "cleaned-val.json")
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "fine_tuned_model"))
os.makedirs(output_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# ---------------------------
# LOAD TOKENISER
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# ---------------------------
# DEFINE MODEL 
# ---------------------------
class WeightedRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.5)  
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = self.dropout(outputs.logits)
        if labels is not None and self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device), label_smoothing=0.1)
            loss = loss_fct(logits, labels)
            outputs = (loss,) + outputs[1:]
        return outputs

# ---------------------------
# DYNAMIC CLASS WEIGHTS
# ---------------------------
class_weights = torch.tensor([1.0, 1.0]).to(device)

model = WeightedRobertaForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2,
    class_weights=class_weights
)
model.to(device)
logging.info("Model loaded with dropout and weighted loss function.")

# ---------------------------
# DATA LOADING & TOKENISATION WITH AUGMENTATION
# ---------------------------
def augment_input(text):
    """Stronger augmentation to prevent overfitting."""
    if random.random() < 0.5:
        text = text.replace("\n", " ")  
    if random.random() < 0.3:
        text = text.replace("Buggy Code:", "Code Snippet:")  
    return text

def load_and_tokenize(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Apply augmentation on each item
    for item in data:
        item["text"] = augment_input(item["text"])
    
    dataset = Dataset.from_list(data)
    
    # Tokenise the dataset
    tokenized = tokenizer(
        dataset["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = dataset["label"]
    tokenized_dataset = Dataset.from_dict(tokenized)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

logging.info("Loading and tokenising training dataset with augmentation...")
combined_train = load_and_tokenize(train_file)
logging.info("Loading and tokenising validation dataset with augmentation...")
combined_val = load_and_tokenize(val_file)

# ---------------------------
# UPDATED TRAINING ARGUMENTS
# ---------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "results"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs"),
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=7,
    learning_rate=2e-5,
    weight_decay=0.02,
    warmup_steps=1000,
    lr_scheduler_type="cosine_with_restarts", 
    load_best_model_at_end=True,
    report_to=None,
    fp16=torch.cuda.is_available()  
)

# ---------------------------
# TRAINING INITIALISATION
# ---------------------------
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
logging.info("Starting model training with improved parameters...")
trainer.train()

# ---------------------------
# SAVE FINAL MODEL
# ---------------------------
model_dir = os.path.join(output_dir, "model_final")
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
logging.info(f"Model saved to {model_dir}")

logging.info("Training complete with improved performance settings.")

