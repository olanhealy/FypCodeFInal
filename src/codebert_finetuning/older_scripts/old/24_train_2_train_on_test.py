# FILE: 23_train_debug_swap_test_train.py

import logging
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("debug_swap_training")
logger.info("DEBUG: Training on TEST set, evaluating on TRAIN set...")

# Load preprocessed JSONL files
data_path = "../../Data/SSTUBS_ENHANCED_23MAR/"
dataset = load_dataset("json", data_files={
    "train": f"{data_path}/train.json",
    "validation": f"{data_path}/val.json",
    "test": f"{data_path}/test.json"
})

# Tokeniser + Preprocessing
model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=False)

logger.info("Tokenising dataset...")
tokenised_dataset = dataset.map(preprocess, batched=True)

# Model + Config
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
for param in model.roberta.parameters():
    param.requires_grad = False
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Training Args
training_args = TrainingArguments(
    output_dir="../../FINAL_MODEL_DEBUG_SWAP",
    learning_rate=2e-7,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=False,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    save_total_limit=1,
    report_to="none",
)

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenised_dataset["test"],     
    eval_dataset=tokenised_dataset["train"],    
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Final Evaluation
logger.info("Final evaluation on swapped TRAIN set...")
metrics = trainer.evaluate(tokenised_dataset["train"])
logger.info(f"Swapped Evaluation Results: {metrics}")

# Save Model
model.save_pretrained("../../FINAL_MODEL_DEBUG_SWAP/model_final")
tokenizer.save_pretrained("../../FINAL_MODEL_DEBUG_SWAP/model_final")
logger.info("Final model saved to: ../../FINAL_MODEL_DEBUG_SWAP/model_final")

