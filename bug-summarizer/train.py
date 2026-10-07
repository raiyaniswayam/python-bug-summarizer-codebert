import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import warnings
warnings.filterwarnings('ignore')

print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Data
data_path = 'data/bug_dataset.csv'
df = pd.read_csv(data_path)

# 2. Tokenize and Map Labels
MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

labels = df['bug_type'].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df['label'] = df['bug_type'].map(label2id)

train_df = df[df['split'] == 'train'].sample(100, random_state=42)
val_df = df[df['split'] == 'val'].sample(20, random_state=42)
test_df = df[df['split'] == 'test'].sample(20, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

def tokenize_function(examples):
    return tokenizer(examples['code'], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])

# 3. Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
).to(device)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 4. Train
print("Training...")
trainer.train()

# 5. Evaluate
print("Evaluating...")
eval_results = trainer.evaluate(tokenized_datasets["test"])
print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Test F1 Score: {eval_results['eval_f1']:.4f}")

# 6. Save
output_dir = "model/codebert_bug_model"
os.makedirs(output_dir, exist_ok=True)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
    json.dump({
        "label_to_id": label2id,
        "id_to_label": id2label
    }, f)

print(f"Model saved to {output_dir}")
