#!/usr/bin/env python3
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
SSTUBS_SPLITS_DIR = os.path.join(BASE_DIR, "Data", "sstubs4j", "unique", "splits")
OUTPUT_DIR = os.path.join(BASE_DIR, "Data", "sstubs_splits_by_project")
os.makedirs(OUTPUT_DIR, exist_ok=True)
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "fine_tuned_model")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
MAX_WORDS = 150

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def truncate_text(text, max_words=150):
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words])

def sanitize_commit_message(msg):
    if not msg:
        return ""
    # Remove explicit markers while preserving technical details
    return msg.lower().replace("buggy", "").replace("fixed", "").replace("bug", "").replace("fix", "")

def build_text_sample(context_before, context_after, snippet, fix_commit_msg, parent_commit_msg):
    # Combine context fields
    full_context = f"{context_before or ''} {context_after or ''}".strip()
    full_context = truncate_text(full_context, MAX_WORDS)
    fix_commit_msg = sanitize_commit_message(fix_commit_msg)
    parent_commit_msg = sanitize_commit_message(parent_commit_msg)
    # Create multi-segment input with explicit separators
    text = (
        f"[CONTEXT] {full_context} [SNIPPET] {truncate_text(snippet, MAX_WORDS)} "
        f"[COMMIT] {fix_commit_msg} [PARENT] {parent_commit_msg}"
    )
    return text.strip()

# ---------------------------
# DATASET CREATION 
# ---------------------------
def create_project_split_dataset():
    logging.info("Loading sstubs4j splits...")
    all_sstubs = []
    for split in ["train", "val", "test"]:
        file_path = os.path.join(SSTUBS_SPLITS_DIR, f"sstubsLarge-{split}.json")
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
                all_sstubs.extend(data)
        else:
            logging.warning(f"File not found: {file_path}")
    logging.info(f"Total samples loaded: {len(all_sstubs)}")
    
    # Group by project
    project_to_samples = defaultdict(list)
    for bug in all_sstubs:
        project = bug.get("projectName", "UNKNOWN_PROJECT")
        project_to_samples[project].append(bug)
    logging.info(f"Unique projects: {len(project_to_samples)}")
    
    # Shuffle projects and split 80/10/10
    projects = list(project_to_samples.keys())
    random.shuffle(projects)
    train_cutoff = int(0.8 * len(projects))
    val_cutoff = int(0.9 * len(projects))
    train_projects = projects[:train_cutoff]
    val_projects = projects[train_cutoff:val_cutoff]
    test_projects = projects[val_cutoff:]
    logging.info(f"Train projects: {len(train_projects)} | Val projects: {len(val_projects)} | Test projects: {len(test_projects)}")
    
    # Build samples for each project list
    def build_samples(project_list):
        samples = []
        for proj in project_list:
            for bug in project_to_samples[proj]:
                context_before = bug.get("contextBefore", "")
                context_after = bug.get("contextAfter", "")
                buggy_code = bug.get("buggyCode", "")
                fixed_code = bug.get("sourceAfterFix", "")
                fix_commit_msg = bug.get("fixCommitMessage", "")
                parent_commit_msg = bug.get("parentCommitMessage", "")
                # Positive sample (buggy)
                samples.append({
                    "text": build_text_sample(context_before, context_after, buggy_code, fix_commit_msg, parent_commit_msg),
                    "label": 1
                })
                # Negative sample (fixed)
                samples.append({
                    "text": build_text_sample(context_before, context_after, fixed_code, fix_commit_msg, parent_commit_msg),
                    "label": 0
                })
        return samples

    train_data = build_samples(train_projects)
    val_data = build_samples(val_projects)
    test_data = build_samples(test_projects)
    
    logging.info(f"Train set size: {len(train_data)} | Val set size: {len(val_data)} | Test set size: {len(test_data)}")
    # Shuffle each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    # Save splits for future reference
    with open(os.path.join(OUTPUT_DIR, "train_sstubs.json"), "w") as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "val_sstubs.json"), "w") as f:
        json.dump(val_data, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "test_sstubs.json"), "w") as f:
        json.dump(test_data, f, indent=4)
    
    logging.info("Saved dataset splits.")
    return train_data, val_data, test_data

# ---------------------------
# MAIN TRAINING PIPELINE
# ---------------------------
def main():
    train_data, val_data, test_data = create_project_split_dataset()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    train_dataset = train_dataset.map(tokenize_fn, batched=True)
    val_dataset = val_dataset.map(tokenize_fn, batched=True)
    train_dataset = train_dataset.rename_column("label", "labels")
    val_dataset = val_dataset.rename_column("label", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Define model
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
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    logging.info("Starting training...")
    trainer.train()
    model_dir = os.path.join(MODEL_OUTPUT_DIR, "model_final")
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    logging.info(f"Model saved to {model_dir}")

if __name__ == "__main__":
    main()

