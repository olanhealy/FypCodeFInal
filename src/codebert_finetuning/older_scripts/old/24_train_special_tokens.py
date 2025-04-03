# FILE: 24_train_with_larger_head.py

import os
import logging
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedModel,
    RobertaConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("larger_head_training")

logger.info("Training with frozen base and larger head...")
# Load Data
data_path = "../../Data/SSTUBS_ENHANCED_23MAR"
dataset = load_dataset("json", data_files={
    "train": f"{data_path}/train.json",
    "validation": f"{data_path}/val.json",
    "test": f"{data_path}/test.json"
})


# Tokeniser & Preprocessing
model_name = "microsoft/codebert-base"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
special_tokens = {
    "additional_special_tokens": ["[CONTEXT]", "[SNIPPET]", "[COMMIT]", "[PARENT]"]
}
tokenizer.add_special_tokens(special_tokens)

def preprocess(example):
    return tokenizer(example["text"], truncation=True, padding=False)

logger.info(" Tokenising dataset...")
tokenised_dataset = dataset.map(preprocess, batched=True)

# Define Custom Model w/ Larger Head
class CustomRobertaClassifier(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.roberta.resize_token_embeddings(len(tokenizer))
        for param in self.roberta.parameters():
            param.requires_grad = False  # Freeze all

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.num_labels)
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

model = CustomRobertaClassifier(RobertaConfig.from_pretrained(model_name, num_labels=2))

# Training Args
training_args = TrainingArguments(
    output_dir="../../FINAL_MODEL_SPECIAL_TOKENS",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    logging_dir="./logs",
    load_best_model_at_end=True,
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
    train_dataset=tokenised_dataset["train"],
    eval_dataset=tokenised_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# Train
logger.info("Training started...")
trainer.train()

# Final Evaluation
logger.info("Evaluating on test set...")
metrics = trainer.evaluate(tokenised_dataset["test"])
logger.info(f"Final Test Results: {metrics}")

# Save
model.save_pretrained("../../FINAL_MODEL_SPECIAL_TOKENS/model_final")
tokenizer.save_pretrained("../../FINAL_MODEL_SPECIAL_TOKENS/model_final")
logger.info("Model saved to: ../../FINAL_MODELSPECIAL_TOKENSD/model_final")

