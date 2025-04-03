import os
import json
import random
import numpy as np

# Force usage of GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ----- Configuration -----
data_files = {
    "train": "../../Data/SSTUBS4J_one/sstubsLarge-train-processed.json",
    "validation": "../../Data/SSTUBS4J_one/sstubsLarge-val-processed.json",
    "test": "../../Data/SSTUBS4J_one/sstubsLarge-test-processed.json"
}

# ----- Load Processed Dataset -----
raw_datasets = load_dataset("json", data_files=data_files)

# ----- Flatten the Dataset -----
def flatten_batch(batch):
    out = {"input_text": [], "label": []}
    for texts, labs in zip(batch["input_text"], batch["label"]):
        for txt, lab in zip(texts, labs):
            out["input_text"].append(txt)
            out["label"].append(lab)
    return out

flattened_train = raw_datasets["train"].map(flatten_batch, batched=True, remove_columns=raw_datasets["train"].column_names)
flattened_val   = raw_datasets["validation"].map(flatten_batch, batched=True, remove_columns=raw_datasets["validation"].column_names)
flattened_test  = raw_datasets["test"].map(flatten_batch, batched=True, remove_columns=raw_datasets["test"].column_names)

# Create a DatasetDict for convenience.
datasets = DatasetDict({
    "train": flattened_train,
    "validation": flattened_val,
    "test": flattened_test
})

print("Flattened dataset sizes:")
print("Train:", len(datasets["train"]))
print("Validation:", len(datasets["validation"]))
print("Test:", len(datasets["test"]))

# ----- Tokenisation -----
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize_function(example):
    return tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["input_text"])
tokenized_datasets.set_format("torch")

# ----- Compute Metrics -----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# ----- Model Setup -----
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
model.config.id2label = {0: "NonBug", 1: "Bug"}
model.config.label2id = {"NonBug": 0, "Bug": 1}

# ----- Training Configuration -----
training_args = TrainingArguments(
    output_dir="./context_aware_bug_localisation_model",
    evaluation_strategy="epoch",    
    save_strategy="epoch",           
    overwrite_output_dir=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    num_train_epochs=3,            
    weight_decay=0.01,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=True,                     
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ----- Training & Evaluation -----
print("Starting training on GPU 0...")
trainer.train()
print("Training complete.")

test_results = trainer.evaluate(tokenized_datasets["test"])
print("Test set evaluation:", test_results)

# Save the fine-tuned model for inference.
trainer.save_model("./context_aware_model_27_localisation_model")

