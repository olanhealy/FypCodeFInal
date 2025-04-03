import os
import json
import torch
import logging
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrain_augmented")
logger.info("Starting retraining with clean negatives...")

# Paths
augmented_train_file = "../../Data/SSTUBS_REP_PREPROCESSED_FINAL/sstubsLarge-train-augmented.json"
val_file = "../../Data/SSTUBS_REP_PREPROCESSED_FINAL/sstubsLarge-val.json"
test_file = "../../Data/SSTUBS_REP_PREPROCESSED_FINAL/sstubsLarge-test.json"

# Load dataset
dataset = load_dataset("json", data_files={
    "train": augmented_train_file,
    "validation": val_file,
    "test": test_file
})

# Tokeniser
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
tokenizer.add_special_tokens({"additional_special_tokens": ["[CONTEXT]", "[SNIPPET]", "[COMMIT]", "[PARENT]"]})

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model from previously trained checkpoint
model_path = "../../FINAL_MODEL/model_final"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
model.resize_token_embeddings(len(tokenizer))
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="../../FINAL_MODEL_AUGMENTED",
    num_train_epochs=2,  
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir="../../FINAL_MODEL_AUGMENTED/logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none",
    fp16=torch.cuda.is_available(),
    overwrite_output_dir=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
logger.info("Training on augmented dataset...")
trainer.train()
logger.info("Training complete.")

# Final save
final_model_dir = "../../FINAL_MODEL_AUGMENTED/model_final"
os.makedirs(final_model_dir, exist_ok=True)
model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
logger.info(f"Final model saved to: {final_model_dir}")

# Evaluate on test
test_results = trainer.evaluate(tokenized_dataset["test"])
logger.info(f"Test Set Performance: {test_results}")

