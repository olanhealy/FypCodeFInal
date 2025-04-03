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
    TrainingArguments,
    EarlyStoppingCallback  
)
import torch.nn as nn
import random

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "final_training_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info(" Starting final training with aggressive anti-overfitting fixes...")

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

# ---------------------------
# LOAD TOKENIZER
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# ---------------------------
# DEFINE MODEL WITH STRONGER REGULARISATION
# ---------------------------
class CustomRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.6)  
        for param in self.roberta.encoder.layer[:8].parameters():  
            param.requires_grad = False  # Freeze 8 out of 12 layers
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = self.dropout(outputs.logits)  
        logits = self.dropout(outputs.logits)  

        if labels is not None:  
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return (loss, logits)  
        return logits  


model = CustomRobertaForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2
)
model.to(device)
logging.info(" Model loaded with stronger regularisation & frozen layers.")

# ---------------------------
# DATA AUGMENTATION 
# ---------------------------
def augment_text(text):
    """Apply noise to text to break memorisation."""
    if random.random() < 0.3:
        text = text.replace("Buggy Code:", "Code Change:")  
    if random.random() < 0.2:
        text = text.replace("\n", " ")  
    return text

def load_and_tokenize(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    
    for item in data:
        item["text"] = augment_text(item["text"])  

    dataset = Dataset.from_list(data)
    tokenized = tokenizer(dataset["text"], truncation=True, padding=True, max_length=512)
    tokenized["labels"] = dataset["label"]
    tokenized_dataset = Dataset.from_dict(tokenized)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset

logging.info(" Loading and tokenising datasets with injected noise...")
train_data = load_and_tokenize(train_file)
val_data = load_and_tokenize(val_file)

# ---------------------------
# TRAINING ARGUMENTS
# ---------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-6,  
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    num_train_epochs=5,  
    weight_decay=0.2,  
    warmup_steps=500,
    lr_scheduler_type="cosine", 
    load_best_model_at_end=True,
    save_total_limit=2,
    fp16=True,  
    logging_steps=100,
    eval_steps=500,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_dir=os.path.join(output_dir, "logs"),
)

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
logging.info(" Starting final training...")
trainer.train()
logging.info(" Training complete with stronger anti-overfitting strategies.")

