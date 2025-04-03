import os
import json
import logging
import torch
import torch.nn as nn
import time
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ---------------------------
# LOGGING CONFIGURATION
# ---------------------------
log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "model_overnight_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logging.info(" Starting Automated Training with Extended Grid Search...")

# ---------------------------
# SET PATHS & CONFIGURATION
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(BASE_DIR, "Data", "sstubs4j_FINAL")
OUTPUT_DIR = os.path.join(BASE_DIR, "fine_tuned_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# FORCE SINGLE-GPU USAGE
# ---------------------------
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD DATASETS
# ---------------------------
def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

train_data = load_json(os.path.join(DATASET_DIR, "sstubsLarge-train.json"))
val_data = load_json(os.path.join(DATASET_DIR, "sstubsLarge-val.json"))

logging.info(f" Train set size: {len(train_data)}")
logging.info(f" Validation set size: {len(val_data)}")

# ---------------------------
# LOAD TOKENISER & MODEL
# ---------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

#  ADD SPECIAL TOKENS
special_tokens = ["[CONTEXT]", "[SNIPPET]", "[COMMIT]", "[PARENT]"]
tokenizer.add_tokens(special_tokens)

class MyRobertaClassifier(RobertaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        self.dropout = nn.Dropout(p=0.3)
        self.class_weights = class_weights.to(device) if class_weights is not None else None

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        logits = self.dropout(outputs.logits)

        if labels is not None and self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights, label_smoothing=0.1)
            loss = loss_fct(logits, labels.to(device))
            outputs = (loss,) + outputs[1:]

        return outputs

class_weights = torch.tensor([1.0, 1.0]).to(device)
model = MyRobertaClassifier.from_pretrained("microsoft/codebert-base", num_labels=2, class_weights=class_weights)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# ---------------------------
# CONVERT DATA TO HUGGING FACE DATASET
# ---------------------------
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ---------------------------
# EXTENDED GRID SEARCH PARAMETERS
# ---------------------------
param_grid = {
    "batch_size": [32, 64],
    "learning_rate": [2e-5, 1e-5, 5e-6],
    "weight_decay": [0.01, 0.001],
    "dropout_rate": [0.3, 0.4],  
    "warmup_steps": [500, 1000],  
    "gradient_accumulation_steps": [1, 2],  
}

best_loss = float("inf")
best_model_path = None

# ---------------------------
# TRAINING WITH GRID SEARCH
# ---------------------------
start_time = time.time()

for batch_size in param_grid["batch_size"]:
    for lr in param_grid["learning_rate"]:
        for wd in param_grid["weight_decay"]:
            for dropout_rate in param_grid["dropout_rate"]:
                for warmup in param_grid["warmup_steps"]:
                    for grad_accum in param_grid["gradient_accumulation_steps"]:

                        logging.info(f" Training with batch={batch_size}, lr={lr}, wd={wd}, dropout={dropout_rate}, warmup={warmup}, grad_accum={grad_accum}...")
                        model.dropout.p = dropout_rate

                        training_args = TrainingArguments(
                            output_dir=os.path.join(OUTPUT_DIR, "results"),
                            evaluation_strategy="epoch",
                            save_strategy="epoch",
                            logging_dir=os.path.join(OUTPUT_DIR, "logs"),

                            per_device_train_batch_size=batch_size,
                            per_device_eval_batch_size=batch_size,
                            gradient_accumulation_steps=grad_accum,

                            num_train_epochs=10,
                            learning_rate=lr,
                            weight_decay=wd,

                            warmup_steps=warmup,
                            lr_scheduler_type="cosine_with_restarts",

                            fp16=True,
                            no_cuda=False,
                            save_total_limit=2,
                            load_best_model_at_end=True,
                            metric_for_best_model="eval_loss",
                            report_to=None
                        )

                        trainer = Trainer(
                            model=model,
                            args=training_args,
                            train_dataset=train_dataset,
                            eval_dataset=val_dataset,
                            tokenizer=tokenizer
                        )

                        trainer.train()

                        # Get best eval loss
                        eval_metrics = trainer.evaluate()
                        eval_loss = eval_metrics["eval_loss"]

                        logging.info(f" Eval loss: {eval_loss}")

                        if eval_loss < best_loss:
                            best_loss = eval_loss
                            best_model_path = os.path.join(OUTPUT_DIR, f"best_model_batch{batch_size}_lr{lr}_wd{wd}_dropout{dropout_rate}_warmup{warmup}_grad{grad_accum}")
                            model.save_pretrained(best_model_path)
                            tokenizer.save_pretrained(best_model_path)
                            logging.info(f"New best model saved at {best_model_path}")

logging.info(" Training finished!")

end_time = time.time()
elapsed_time = (end_time - start_time) / 3600
logging.info(f"Total training time: {elapsed_time:.2f} hours")

logging.info(f"Best model found at {best_model_path} with eval loss {best_loss}")

