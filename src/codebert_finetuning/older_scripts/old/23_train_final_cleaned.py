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

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("final_cleaned_training")
logger.info("Starting FINAL training run with cleaned and deduplicated repetition data")

# Load preprocessed JSON files
data_path = "../../Data/SSTUBS_ENHANCED_23MAR/"
dataset = load_dataset("json", data_files={
    "train": f"{data_path}/train_augmented.json",
    "validation": f"{data_path}/val.json",
    "test": f"{data_path}/test.json"
})

# Tokeniser + Preprocessing
model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=False)

logger.info(" Tokenising dataset...")
tokenised_dataset = dataset.map(preprocess, batched=True)

# Model + Config
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Training Args
training_args = TrainingArguments(
    output_dir="../../FINAL_MODEL_23MAR",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    save_total_limit=1,
    report_to="none",
)

# Metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
    train_dataset=tokenised_dataset["train"],
    eval_dataset=tokenised_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# Train!
logger.info("Training on cleaned + masked repetition dataset...")
trainer.train()

# Final Test Eval
logger.info(" Evaluating on test set...")
metrics = trainer.evaluate(tokenised_dataset["test"])
logger.info(f"Final Test Results: {metrics}")

# Save Model
model.save_pretrained("../../FINAL_MODEL_23MAR/model_final")
tokenizer.save_pretrained("../../FINAL_MODEL_23MAR/model_final")
logger.info("Final model saved to: ../../FINAL_MODEL_23MAR/model_final")

