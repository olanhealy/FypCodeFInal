import os
import json
import logging
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "model_22_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logging.info(" Starting Optimised Training...")

# ---------------------------
# LOAD DATASET
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "Data", "sstubs4j_FINAL")  
OUTPUT_DIR = os.path.join(BASE_DIR, "fine_tuned_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(file_path):
    """Load JSON dataset."""
    with open(file_path, "r") as f:
        return json.load(f)

train_data = load_json(os.path.join(DATASET_DIR, "sstubsLarge-train.json"))
val_data = load_json(os.path.join(DATASET_DIR, "sstubsLarge-val.json"))

# CHECK CLASS BALANCE
from collections import Counter
labels = [example["label"] for example in train_data]
logging.info(f" Label Distribution: {Counter(labels)}")  

# ---------------------------
# TOKENIZER & MODEL
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
special_tokens = ["[CONTEXT]", "[SNIPPET]", "[COMMIT]", "[PARENT]"]
tokenizer.add_tokens(special_tokens) 

class MyRobertaClassifier(RobertaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.3)
        self.class_weights = class_weights.to(device) if class_weights is not None else None

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = self.dropout(outputs.logits)

        if labels is not None and self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
            loss = loss_fct(logits, labels.to(device))  
            outputs = (loss,) + outputs[1:]

        return outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor([1.0, 1.0]).to(device)  
model = MyRobertaClassifier.from_pretrained("microsoft/codebert-base", num_labels=2, class_weights=class_weights)
model.resize_token_embeddings(len(tokenizer))  #  Resize model for new tokens
model.to(device)  

# ---------------------------
# DATASET PREPROCESSING
# ---------------------------
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------------------
# TRAINING CONFIGURATION
# ---------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "results"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,   
    gradient_accumulation_steps=2,  
    num_train_epochs=10,  
    learning_rate=5e-6,  
    weight_decay=0.01,
    warmup_steps=500,   
    lr_scheduler_type="cosine_with_restarts",  
    fp16=True,  
    no_cuda=False,  
    load_best_model_at_end=True,  
    report_to=None  
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

logging.info(" Starting Optimised Training...")
trainer.train()

# ---------------------------
# SAVE MODEL
# ---------------------------
model_dir = os.path.join(OUTPUT_DIR, "model_final")
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
logging.info(f" Model saved to {model_dir}")

