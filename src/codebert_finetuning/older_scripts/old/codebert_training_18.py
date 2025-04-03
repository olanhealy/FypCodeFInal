import os
import json
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
sstubs_repetition_path = os.path.join("..", "..", "Data", "sstubs4j", "repetition", "splits")
defects4j_path      = os.path.join("..", "..", "Data", "defects4j", "splits")

output_dir = os.path.join("..", "..", "fine_tuned_model")
os.makedirs(output_dir, exist_ok=True)

# Set CUDA device for utilising Nvidia A100 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD TOKENISER
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# ---------------------------
# FUNCTIONS FOR SSTUBS4J DATA (UNIQUE & REPETITION)
# ---------------------------
def load_sstubs(data_path):
    """Load train, validation, and test JSON files from sstubs4j data path."""
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
      - Buggy example (label 1): uses ONLY the contextBefore field along with
        the commit messages, bug type, and project name.
      - Fixed example (label 0): uses ONLY the contextAfter field along with
        the commit messages, bug type, and project name.
    NOTE: While preprecessing this dataset, I added in "NO DATA AVAILABlE
    For Commit messages if  empty string, however since doing some training 
    I am changing this to be empty strings as model may assocaite this with actual bug
    """
    inputs = []
    labels = []
    for bug in tqdm(data, desc="Preprocessing sstubs4j data"):
        context_before = bug.get("contextBefore", "")
        context_after = bug.get("contextAfter", "")
        # Extract commit messages and replace "NO DATA AVAILABLE" if there
        fix_commit_message = bug.get("fixCommitMessage", "")
        if fix_commit_message.strip().upper() == "NO DATA AVAILABLE":
            fix_commit_message = ""
        parent_commit_msg = bug.get("parentCommitMessage", "")
        if parent_commit_msg.strip().upper() == "NO DATA AVAILABLE":
            parent_commit_msg = ""
        bug_type = bug.get("bugType", "")
        project_name = bug.get("projectName", "")
        
        # Construct buggy example (label 1) 
        buggy_text = (
            f"Context Before:\n{context_before}\n"
            f"Fix Commit Message: {fix_commit_message}\n"
            f"Parent Commit Message: {parent_commit_msg}\n"
            f"Bug Type: {bug_type}\n"
            f"Project: {project_name}"
        )
        inputs.append(buggy_text)
        labels.append(1)
        
        # Construct fixed example (label 0) 
        fixed_text = (
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

def prepare_sstubs(data_path):
    train_data, val_data, test_data = load_sstubs(data_path)
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
    For defects4j, since it only includes the bug, just creating one example per bug (label 1) using:
      - Git Diff, Failing Tests, and Repair Patterns.
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
unique_train, unique_val, unique_test = prepare_sstubs(sstubs_unique_path)
logging.info("sstubs4j UNIQUE dataset loaded.")

logging.info("Loading and preparing sstubs4j REPETITION dataset...")
repetition_train, repetition_val, repetition_test = prepare_sstubs(sstubs_repetition_path)
logging.info("sstubs4j REPETITION dataset loaded.")

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
# PHASE 1: TRAIN ON SSTUBS4J UNIQUE DATASET 
# ---------------------------
best_lr = 2e-05
best_bs = 8
num_epochs = 3

training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "results_unique_best"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs_unique_best"),
    per_device_train_batch_size=best_bs,
    per_device_eval_batch_size=best_bs,
    num_train_epochs=num_epochs,
    learning_rate=best_lr,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to=None
)

# Reinitialise a fresh model
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

logging.info("Training UNIQUE model using best hyperparameters (lr=2e-05, bs=8)...")
trainer.train()
unique_eval_results = trainer.evaluate(unique_val)
logging.info(f"UNIQUE evaluation results: {unique_eval_results}")

# Save the model from UNIQUE training
model_dir_unique = os.path.join(output_dir, "model_unique")
if os.path.exists(model_dir_unique):
    shutil.rmtree(model_dir_unique)
os.makedirs(model_dir_unique, exist_ok=True)
model.save_pretrained(model_dir_unique)
tokenizer.save_pretrained(model_dir_unique)
logging.info(f"Model from UNIQUE training saved in: {model_dir_unique}")

# ---------------------------
# PHASE 2: FINE-TUNE ON SSTUBS4J REPETITION DATASET
# ---------------------------
logging.info("Loading model for repetition fine-tuning...")
model = RobertaForSequenceClassification.from_pretrained(model_dir_unique, num_labels=2)
model.to(device)

repetition_training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "results_repetition"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(output_dir, "logs_repetition"),
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to=None
)

repetition_trainer = Trainer(
    model=model,
    args=repetition_training_args,
    train_dataset=repetition_train,
    eval_dataset=repetition_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
logging.info("Fine-tuning on repetition dataset...")
repetition_trainer.train()
repetition_eval_results = repetition_trainer.evaluate(repetition_val)
logging.info(f"Repetition evaluation results: {repetition_eval_results}")

# Save the model from REPETITION fine-tuning
model_dir_repetition = os.path.join(output_dir, "model_repetition")
if os.path.exists(model_dir_repetition):
    shutil.rmtree(model_dir_repetition)
os.makedirs(model_dir_repetition, exist_ok=True)
model.save_pretrained(model_dir_repetition)
tokenizer.save_pretrained(model_dir_repetition)
logging.info(f"Model from REPETITION fine-tuning saved in: {model_dir_repetition}")

# ---------------------------
# PHASE 3: FINE-TUNE ON DEFECTS4J DATASET
# ---------------------------
logging.info("Loading model for defects4j fine-tuning...")
model = RobertaForSequenceClassification.from_pretrained(model_dir_repetition, num_labels=2)
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

logging.info("Evaluating final model on REPETITION test set...")
final_repetition_results = defects_trainer.evaluate(repetition_test)
logging.info(f"Final REPETITION test set results: {final_repetition_results}")

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

