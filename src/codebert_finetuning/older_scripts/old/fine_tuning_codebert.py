import os
import json
import random
import itertools
import shutil
import torch
import logging
import sys
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "integrated_training_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.info("Starting integrated training script...")

# ---------------------------
# SET PATHS & ENVIRONMENT
# ---------------------------
# Define dataset paths
sstubs_unique_path = os.path.join("..", "..", "Data", "sstubs4j", "unique", "splits")
defects4j_path      = os.path.join("..", "..", "Data", "defects4j", "splits")

# Output directory for models and results
output_dir = os.path.join("..", "..", "fine_tuned_model")
os.makedirs(output_dir, exist_ok=True)

# Set CUDA device for utlising Nivida A100 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD TOKENISER
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# ---------------------------
# FUNCTIONS FOR SSTUBS4J UNIQUE DATASET
# ---------------------------
def load_sstubs_unique(data_path):
    """Load train, validation, and test JSON files for sstubs4j unique."""
    train_path = os.path.join(data_path, "sstubsLarge-train.json")
    val_path   = os.path.join(data_path, "sstubsLarge-val.json")
    test_path  = os.path.join(data_path, "sstubsLarge-test.json")
    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(val_path, "r") as f:
        val_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)
    return train_data, val_data, test_data

def preprocess_sstubs(data):
    """
    For each bug in sstubs4j UNIQUE, I created two examples for training: 
      - Buggy example (label 1): includes Context Before, Buggy Code,
        Fix Commit Message, Parent Commit Message, Bug Type, and Project Name.
      - Fixed example (label 0): includes Context Before, Context After,
        Fix Commit Message, Parent Commit Message, Bug Type, and Project Name
    """
    inputs = []
    labels = []
    for bug in tqdm(data, desc="Preprocessing sstubs4j unique data"):
        # Extract fields from JSON
        context_before     = bug.get("contextBefore", "")
        buggy_code         = bug.get("buggyCode", "")
        context_after      = bug.get("contextAfter", "")
        fix_commit_message = bug.get("fixCommitMessage", "")
        parent_commit_msg  = bug.get("parentCommitMessage", "")
        bug_type           = bug.get("bugType", "")
        project_name       = bug.get("projectName", "")
        
        # Buggy example (label 1)
        buggy_text = (
            f"Context Before:\n{context_before}\n"
            f"Buggy Code:\n{buggy_code}\n"
            f"Fix Commit Message: {fix_commit_message}\n"
            f"Parent Commit Message: {parent_commit_msg}\n"
            f"Bug Type: {bug_type}\n"
            f"Project: {project_name}"
        )
        inputs.append(buggy_text)
        labels.append(1)
        
        # Fixed example (label 0)
        fixed_text = (
            f"Context Before:\n{context_before}\n"
            f"Context After:\n{context_after}\n"
            f"Fix Commit Message: {fix_commit_message}\n"
            f"Parent Commit Message: {parent_commit_msg}\n"
            f"Bug Type: {bug_type}\n"
            f"Project: {project_name}"
        )
        inputs.append(fixed_text)
        labels.append(0)
    tokenized = tokenizer(inputs, truncation=True, padding=True, max_length=512)
    tokenized["labels"] = labels
    ds = Dataset.from_dict(tokenized)
    ds.set_format("torch")
    return ds

def prepare_sstubs_unique(data_path):
    train_data, val_data, test_data = load_sstubs_unique(data_path)
    train_ds = preprocess_sstubs(train_data)
    val_ds   = preprocess_sstubs(val_data)
    test_ds  = preprocess_sstubs(test_data)
    return train_ds, val_ds, test_ds

# ---------------------------
# FUNCTIONS FOR DEFECTS4J DATASET
# ---------------------------
def load_defects4j(data_path):
    """Load train, validation, and test JSON files for defects4j."""
    train_path = os.path.join(data_path, "defects4j-train.json")
    val_path   = os.path.join(data_path, "defects4j-val.json")
    test_path  = os.path.join(data_path, "defects4j-test.json")
    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(val_path, "r") as f:
        val_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)
    return train_data, val_data, test_data

def preprocess_defects4j(data):
    """
    For defects4j, since it only includes the bug, I just create one label of buggy code with the following fields:
      -Git Diff, Failing Tests, and Repair Patterns.
    """
    inputs = []
    labels = []
    for bug in tqdm(data, desc="Preprocessing defects4j data"):
        text = (
            f"Git Diff:\n{bug.get('diff', '')}\n"
            f"Failing Tests:\n{json.dumps(bug.get('failingTests', []), indent=2)}\n"
            f"Repair Patterns: {', '.join(bug.get('repairPatterns', []))}"
        )
        inputs.append(text)
        labels.append(1)
    tokenized = tokenizer(inputs, truncation=True, padding=True, max_length=512)
    tokenized["labels"] = labels
    ds = Dataset.from_dict(tokenized)
    ds.set_format("torch")
    return ds

def prepare_defects4j(data_path):
    train_data, val_data, test_data = load_defects4j(data_path)
    train_ds = preprocess_defects4j(train_data)
    val_ds   = preprocess_defects4j(val_data)
    test_ds  = preprocess_defects4j(test_data)
    return train_ds, val_ds, test_ds

# ---------------------------
# LOAD AND PREPARE DATASETS
# ---------------------------
logging.info("Loading and preparing sstubs4j UNIQUE dataset...")
unique_train, unique_val, unique_test = prepare_sstubs_unique(sstubs_unique_path)
logging.info("sstubs4j UNIQUE dataset loaded.")

logging.info("Loading and preparing DEFECTS4J dataset...")
defects_train, defects_val, defects_test = prepare_defects4j(defects4j_path)
logging.info("Defects4j dataset loaded.")

# ---------------------------
# DEFINE METRICS FUNCTION
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {"eval_accuracy": acc, "eval_precision": precision, "eval_recall": recall, "eval_f1": f1}

# ---------------------------
# PHASE 1: GRID SEARCH ON SSTUBS4J UNIQUE DATASET
# ---------------------------
# Define hyperparameter grid
learning_rates = [2e-5, 1e-5, 3e-5]
batch_sizes = [8, 16]
num_epochs = 3

results = [] 

# Track best model 
best_f1 = -float("inf")
best_config = None
best_model_unique_dir = os.path.join(output_dir, "best_model_unique")
if os.path.exists(best_model_unique_dir):
    shutil.rmtree(best_model_unique_dir)
os.makedirs(best_model_unique_dir, exist_ok=True)

logging.info("Starting grid search on sstubs4j UNIQUE dataset...")
for lr, bs in itertools.product(learning_rates, batch_sizes):
    config_dir = os.path.join(output_dir, f"results_unique_lr_{lr}_bs_{bs}")
    logging.info(f"Training UNIQUE model with lr={lr}, batch size={bs}...")
    
    training_args = TrainingArguments(
        output_dir=config_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(config_dir, "logs"),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=None
    )
    
    # Reinitialise a fresh model for this configuration
    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
    model.to(device)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=unique_train,
        eval_dataset=unique_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    eval_results = trainer.evaluate(unique_val)
    logging.info(f"UNIQUE results for lr={lr}, bs={bs}: {eval_results}")
    results.append((lr, bs, eval_results))
    
    current_f1 = eval_results.get("eval_f1", 0)
    logging.info(f"Configuration metrics: F1={current_f1:.4f}")
    
    if current_f1 > best_f1:
        logging.info(f"New best UNIQUE model found: F1 improved from {best_f1:.4f} to {current_f1:.4f}.")
        best_f1 = current_f1
        best_config = (lr, bs)
        if os.path.exists(best_model_unique_dir):
            shutil.rmtree(best_model_unique_dir)
        shutil.copytree(config_dir, best_model_unique_dir)
    else:
        logging.info(f"UNIQUE model with lr={lr}, bs={bs} did not improve best F1; deleting its directory.")
        shutil.rmtree(config_dir)

results_file_unique = os.path.join(output_dir, "unique_hyperparameter_results.txt")
with open(results_file_unique, "w") as f:
    for lr, bs, res in results:
        f.write(f"lr: {lr}, bs: {bs} => {res}\n")
logging.info("Grid search on UNIQUE dataset complete.")
logging.info(f"Best UNIQUE config: lr={best_config[0]}, bs={best_config[1]}, F1={best_f1:.4f}")
logging.info(f"Best UNIQUE model saved in: {best_model_unique_dir}")

# ---------------------------
# PHASE 2: FINE-TUNE ON DEFECTS4J DATASET
# ---------------------------
logging.info("Loading best UNIQUE model for defects4j fine-tuning...")
model = RobertaForSequenceClassification.from_pretrained(best_model_unique_dir, num_labels=2)
model.to(device)

defects_training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "results_defects4j"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs_defects4j"),
    num_train_epochs=2, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to=None
)

defects_trainer = Trainer(
    model=model,
    args=defects_training_args,
    train_dataset=defects_train,
    eval_dataset=defects_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
logging.info("Fine-tuning on defects4j dataset...")
defects_trainer.train()
defects_eval_results = defects_trainer.evaluate(defects_val)
logging.info(f"Defects4j evaluation results: {defects_eval_results}")

# ---------------------------
# FINAL EVALUATION ON TEST SETS
# ---------------------------
logging.info("Evaluating final model on UNIQUE test set...")
final_unique_results = defects_trainer.evaluate(unique_test)
logging.info(f"Final UNIQUE test set results: {final_unique_results}")

logging.info("Evaluating final model on DEFECTS4J test set...")
final_defects_results = defects_trainer.evaluate(defects_test)
logging.info(f"Final DEFECTS4J test set results: {final_defects_results}")

# ---------------------------
# SAVE FINAL MODEL
# ---------------------------
final_model_dir = os.path.join(output_dir, "final_model")
if os.path.exists(final_model_dir):
    shutil.rmtree(final_model_dir)
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
logging.info(f"Final model saved in: {final_model_dir}")

logging.info("Integrated training complete.")

