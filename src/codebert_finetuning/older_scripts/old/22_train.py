import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------- LOAD DATA --------
data_path = "../../Data/SSTUBS_ENHANCED_AUG"
dataset = load_dataset("json", data_files={
    "train": os.path.join(data_path, "train.json"),
    "validation": os.path.join(data_path, "val.json"),
    "test": os.path.join(data_path, "test.json"),
})

# -------- TOKENIZER --------
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
tokenizer.add_special_tokens({"additional_special_tokens": ["[CONTEXT]", "[SNIPPET]", "[COMMIT]", "[PARENT]"]})

def tokenize_fn(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# -------- METRICS --------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# -------- MODEL --------
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)
model.resize_token_embeddings(len(tokenizer))
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# -------- TRAINING ARGS --------
training_args = TrainingArguments(
    output_dir="../../FINAL_MODEL",
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=1,
    logging_dir="../../FINAL_MODEL/logs",
    report_to="none",
    label_smoothing_factor=0.1,
    fp16=torch.cuda.is_available(),
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()
print(" Training complete.")

# -------- SAVE FINAL --------
model.save_pretrained("../../FINAL_MODEL/model_final")
tokenizer.save_pretrained("../../FINAL_MODEL/model_final")
print(" Model saved to ../../FINAL_MODEL/model_final")

