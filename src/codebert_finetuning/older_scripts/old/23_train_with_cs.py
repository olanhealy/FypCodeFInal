import os
import json
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === Configs ===
BASE_MODEL_DIR = "../../FINAL_MODEL_MERGED/model_final"  # Previous trained model
DATA_DIR = "../../Data/SSTUBS_ENHANCED_23MAR"
TRAIN_FILE = f"{DATA_DIR}/train-merged-with-codesearch.json"  # Merged dataset
VAL_FILE = f"{DATA_DIR}/val.json"
TEST_FILE = f"{DATA_DIR}/test.json"
OUTPUT_DIR = "../../FINAL_MODEL_MERGED_PLUS_CLEAN"
LOGGING_DIR = os.path.join(OUTPUT_DIR, "logs")

# === Dataset Loading ===
def load_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def preprocess(dataset, tokenizer):
    def encode(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)
    return dataset.map(encode, batched=True)

print("Loading datasets...")
tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL_DIR)

train_dataset = preprocess(Dataset.from_list(load_json_file(TRAIN_FILE)), tokenizer)
val_dataset = preprocess(Dataset.from_list(load_json_file(VAL_FILE)), tokenizer)
test_dataset = preprocess(Dataset.from_list(load_json_file(TEST_FILE)), tokenizer)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# === Load Previous Model ===
print("Loading previous model from:", BASE_MODEL_DIR)
model = RobertaForSequenceClassification.from_pretrained(BASE_MODEL_DIR, num_labels=2)

# === Training Args ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir=LOGGING_DIR,
    logging_strategy="steps",
    logging_steps=100,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    warmup_steps=500,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none"
)

# === Metrics ===
def compute_metrics(p):
    preds = torch.argmax(torch.tensor(p.predictions), axis=1)
    labels = torch.tensor(p.label_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# === Start Training ===
print("Fine-tuning on merged dataset...")
trainer.train()

# === Save Final Model ===
final_model_dir = os.path.join(OUTPUT_DIR, "model_final")
trainer.save_model(final_model_dir)
print(f"Final model saved to: {final_model_dir}")

# === Evaluate on Test Set ===
print("Evaluating on test set...")
metrics = trainer.evaluate(eval_dataset=test_dataset)
print("Test Set Metrics:", metrics)

