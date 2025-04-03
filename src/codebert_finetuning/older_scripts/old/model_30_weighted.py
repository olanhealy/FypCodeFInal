import os
import json
import torch
import logging
import numpy as np
from collections import Counter
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "model_30_log_weighted.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("Starting training script with class weighting: model_30_weighted.py")

# ---------------------------
# SET PATHS & LOAD PROCESSED DATASETS
# ---------------------------
processed_dir = "../../Data/SSTUBS_REP_PREPROCESSED_FINAL"
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
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
special_tokens = {"additional_special_tokens": ["[CONTEXT]", "[SNIPPET]", "[COMMIT]", "[PARENT]"]}
tokenizer.add_special_tokens(special_tokens)

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

logging.info("Tokenising dataset...")
tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ---------------------------
# COMPUTE CLASS WEIGHTS
# ---------------------------
logging.info("Computing class weights from training data...")
train_labels = [example["label"] for example in dataset["train"]]
label_counts = Counter(train_labels)
logging.info(f"Label counts: {label_counts}")

total = sum(label_counts.values())
class_weights = torch.tensor([2.0, 1.0])
weights_tensor = torch.tensor(class_weights)
logging.info(f"Computed class weights: {weights_tensor}")

# ---------------------------
# CUSTOM MODEL CLASS
# ---------------------------
class WeightedRoberta(nn.Module):
    def __init__(self, base_model, class_weights):
        super().__init__()
        self.model = base_model
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}

# ---------------------------
# LOAD BASE MODEL
# ---------------------------
base_model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
base_model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = WeightedRoberta(base_model, weights_tensor.to(device))
model.to(device)

# ---------------------------
# METRICS
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ---------------------------
# TRAINING CONFIG
# ---------------------------
training_args = TrainingArguments(
    output_dir="../../fine_tuned_model_weighted",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine_with_restarts",
    logging_dir="../../fine_tuned_model_weighted/logs",
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
# TRAINING
# ---------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

logging.info("Starting training...")
trainer.train()
logging.info("Training complete.")

# ---------------------------
# TEST EVALUATION
# ---------------------------
test_results = trainer.evaluate(tokenized_dataset["test"])
logging.info(f"Test set evaluation: {test_results}")

# ---------------------------
# SAVE FINAL MODEL & TOKENISER
# ---------------------------
final_model_dir = "../../fine_tuned_model_weighted/model_final/"
os.makedirs(final_model_dir, exist_ok=True)
base_model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)
logging.info(f"Model saved to {final_model_dir}")
