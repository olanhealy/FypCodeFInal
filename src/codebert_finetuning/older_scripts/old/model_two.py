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
log_file = os.path.join(log_dir, "updated_training_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("Starting updated training script...")

# ---------------------------
# SET PATHS & ENVIRONMENT
# ---------------------------
sstubs_unique_path = os.path.join("..", "..", "Data", "sstubs4j", "unique", "splits")
sstubs_repetition_path = os.path.join("..", "..", "Data", "sstubs4j", "repetition", "splits")
output_dir = os.path.join("..", "..", "fine_tuned_model")
os.makedirs(output_dir, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD TOKENISER
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# ---------------------------
# DEFINE MODEL WITH DROPOUT & WEIGHTED LOSS
# ---------------------------
class WeightedRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.3)
        self.class_weights = class_weights

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = self.dropout(outputs.logits)
        if labels is not None and self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
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
# SIMPLE DATA AUGMENTATION FUNCTION
# ---------------------------
def augment_input(text):
    """
    Remove non-essential fields (like Project) to mimic real pre-commit input.
    Currently, it drops any text after the 'Project:' marker.
    """
    if "Project:" in text:
        text = text.split("Project:")[0].strip()
    return text

# ---------------------------
# PREPROCESSING FUNCTION FOR SSTUBS4J DATASET (WITHOUT Project)
# ---------------------------
def load_and_preprocess_sstubs(data_path):
    train_path = os.path.join(data_path, "sstubsLarge-train.json")
    val_path = os.path.join(data_path, "sstubsLarge-val.json")
    test_path = os.path.join(data_path, "sstubsLarge-test.json")

    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(val_path, "r") as f:
        val_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)

    return preprocess_sstubs(train_data), preprocess_sstubs(val_data), preprocess_sstubs(test_data)

def preprocess_sstubs(data):
    buggy_samples = []
    not_buggy_samples = []

    for bug in tqdm(data, desc="Preprocessing sstubs4j data"):
        context_before = bug.get("contextBefore", "")
        context_after = bug.get("contextAfter", "")
        buggy_code = bug.get("buggyCode", "")
        source_after_fix = bug.get("sourceAfterFix", "")
        fix_commit_message = bug.get("fixCommitMessage", "")
        parent_commit_msg = bug.get("parentCommitMessage", "")
        bug_type = bug.get("bugType", "")
        # Removed project name from the input since it isn't relevant

        # Construct the input text for buggy (Label 1) and fixed (Label 0) examples.
        buggy_text = (
            f"Context Before:\n{context_before}\n"
            f"Buggy Code: {buggy_code}\n"
            f"Commit Message: {fix_commit_message}\n"
            f"Parent Commit: {parent_commit_msg}\n"
            f"Bug Type: {bug_type}"
        )
        not_buggy_text = (
            f"Context After:\n{context_after}\n"
            f"Fixed Code: {source_after_fix}\n"
            f"Commit Message: {fix_commit_message}\n"
            f"Parent Commit: {parent_commit_msg}\n"
            f"Bug Type: {bug_type}"
        )
        
        # Apply augmentation to simulate variability (e.g., dropping non-essential info)
        buggy_text = augment_input(buggy_text)
        not_buggy_text = augment_input(not_buggy_text)
        
        buggy_samples.append(buggy_text)
        not_buggy_samples.append(not_buggy_text)

    # Balance the dataset by taking the minimum number of samples between the two classes.
    min_samples = min(len(buggy_samples), len(not_buggy_samples))
    buggy_samples = buggy_samples[:min_samples]
    not_buggy_samples = not_buggy_samples[:min_samples]

    inputs = buggy_samples + not_buggy_samples
    labels = [1] * len(buggy_samples) + [0] * len(not_buggy_samples)

    # Log dataset statistics
    logging.info(f"Final dataset contains {len(buggy_samples)} Buggy samples and {len(not_buggy_samples)} Not Buggy samples.")

    # Tokenisation
    tokenized = tokenizer(inputs, truncation=True, padding=True, max_length=512)
    tokenized["labels"] = labels
    ds = Dataset.from_dict(tokenized)
    ds.set_format("torch")

    return ds

# ---------------------------
# LOAD DATASETS
# ---------------------------
unique_train, unique_val, unique_test = load_and_preprocess_sstubs(sstubs_unique_path)
repetition_train, repetition_val, repetition_test = load_and_preprocess_sstubs(sstubs_repetition_path)

logging.info("Datasets loaded and preprocessed.")

# ---------------------------
# UPDATED TRAINING ARGUMENTS
# ---------------------------
training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "results"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs"),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to=None
)

# ---------------------------
# TRAINING INITIALISATION
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=unique_train,
    eval_dataset=unique_val,
    tokenizer=tokenizer,
)

# ---------------------------
# START TRAINING
# ---------------------------
logging.info("Starting model training...")
trainer.train()

# ---------------------------
# SAVE FINAL MODEL
# ---------------------------
model_dir = os.path.join(output_dir, "model_final")
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
logging.info(f"Model saved to {model_dir}")

logging.info("Training complete.")

