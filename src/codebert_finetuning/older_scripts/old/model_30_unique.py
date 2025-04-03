import os
import json
import torch
import logging
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "model_30_unique.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("Starting training script: model_30.py")

# ---------------------------
# SET PATHS & LOAD PROCESSED DATASETS
# ---------------------------
processed_dir = "../../Data/SSTUBS_UNIQUE_PREPROCESSED_FINAL"
train_file = os.path.join(processed_dir, "sstubsLarge-train.json")
val_file   = os.path.join(processed_dir, "sstubsLarge-val.json")
test_file  = os.path.join(processed_dir, "sstubsLarge-test.json")

logging.info("Loading processed dataset...")
dataset = load_dataset("json", data_files={
    "train": train_file,
    "validation": val_file,
    "test": test_file
})

# ---------------------------
# TOKENISATION
# ---------------------------
# Load CodeBERT tokenizer and add special tokens 
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
special_tokens = {"additional_special_tokens": ["[CONTEXT]", "[SNIPPET]", "[COMMIT]", "[PARENT]"]}
tokenizer.add_special_tokens(special_tokens)

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

logging.info("Tokenising dataset...")
tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ---------------------------
# DEFINE EVALUATION METRICS
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ---------------------------
# LOAD CODEBERT MODEL FOR CLASSIFICATION
# ---------------------------
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------------------
# TRAINING CONFIGURATION
# ---------------------------
training_args = TrainingArguments(
    output_dir="../../fine_tuned_model_unique",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine_with_restarts",
    logging_dir="../../fine_tuned_model/logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
    overwrite_output_dir=True
)

# ---------------------------
# INITIALISE TRAINER
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

logging.info("Starting training on GPU 0...")
trainer.train()
logging.info("Training complete.")

test_results = trainer.evaluate(tokenized_dataset["test"])
logging.info(f"Test set evaluation: {test_results}")

# ---------------------------
# SAVE FINAL MODEL & TOKENIZER
# ---------------------------
final_model_dir = "../../fine_tuned_model_unique/model_final/"
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
logging.info(f"Model saved to {final_model_dir}")

