import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Section 1
cells.append(nbf.v4.new_markdown_cell("""# 🐛 Python Source Code Bug Summarizer using CodeBERT (Hybrid Pipeline)

**NLP Assignment — Semester 7**

---

## 1. Introduction and problem statement
This notebook fine-tunes **CodeBERT** (`microsoft/codebert-base`) to act as the ML component of our hybrid bug classifier.
While static tools (AST/Pyflakes) and restricted dynamic execution handle explicit errors (Syntax, Indentation, Undefined Variables), CodeBERT is tasked with finding **logical bugs** such as `off_by_one`, `wrong_comparison`, and `missing_return`.

The pipeline combines these to handle a wide range of Python bugs.
"""))

# Section 2
cells.append(nbf.v4.new_markdown_cell("## 2. Install libraries and check the GPU"))
cells.append(nbf.v4.new_code_cell("""import torch
import sys

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("WARNING: No GPU found. Training on CPU will be slow.")
    device = torch.device("cpu")

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
print("Libraries imported successfully!")"""))

# Section 3
cells.append(nbf.v4.new_markdown_cell("## 3. Load and explore the dataset"))
cells.append(nbf.v4.new_code_cell("""data_path = '../data/bug_dataset.csv'
if not os.path.exists(data_path):
    data_path = 'data/bug_dataset.csv'
df = pd.read_csv(data_path)
print(f"Dataset size: {df.shape}")
display(df.head())

# Class distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='bug_type', data=df, order=df['bug_type'].value_counts().index)
plt.title("Distribution of Bug Types")
plt.show()"""))

# Section 4
cells.append(nbf.v4.new_markdown_cell("## 4. Preprocess and tokenize"))
cells.append(nbf.v4.new_code_cell("""MODEL_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Create label mapping
labels = df['bug_type'].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
print("Labels:", label2id)

# Apply mapping
df['label'] = df['bug_type'].map(label2id)

# Split dataset
train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['code'], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch", columns=['input_ids', 'attention_mask', 'label'])
print("Tokenization complete.")"""))

# Section 5
cells.append(nbf.v4.new_markdown_cell("## 5. Load CodeBERT and fine-tune"))
cells.append(nbf.v4.new_code_cell("""model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
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
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()"""))

# Section 6
cells.append(nbf.v4.new_markdown_cell("## 6. Evaluate the results on Test Set"))
cells.append(nbf.v4.new_code_cell("""# Evaluate on test set
eval_results = trainer.evaluate(tokenized_datasets["test"])
print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Test F1 Score: {eval_results['eval_f1']:.4f}")

# Get predictions
predictions = trainer.predict(tokenized_datasets["test"])
preds = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

print("\\nClassification Report:")
print(classification_report(y_true, preds, target_names=labels))

# Confusion Matrix
cm = confusion_matrix(y_true, preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()"""))

# Section 7
cells.append(nbf.v4.new_markdown_cell("## 7. Save Model"))
cells.append(nbf.v4.new_code_cell("""import os

output_dir = "../model/codebert_bug_model"
if not os.path.exists("../model"):
    output_dir = "model/codebert_bug_model"
os.makedirs(output_dir, exist_ok=True)

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

# Save label mapping
with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
    json.dump({
        "label_to_id": label2id,
        "id_to_label": id2label
    }, f)

print(f"Model and tokenizer saved to {output_dir}")"""))

# Section 8
cells.append(nbf.v4.new_markdown_cell("""## 8. Limitations

The hybrid approach correctly mitigates the weaknesses of ML-only bug detection by deferring exact errors (syntax/runtime) to standard tools. However, there are still limitations to this system:
1. **Intent-based Bugs**: The system cannot detect bugs where the code successfully compiles and executes logic flawlessly, but the logic simply does not match the developer's abstract intent (e.g. implementing a BFS instead of a DFS).
2. **Context constraints**: CodeBERT operates on a 256-token limit here. Bugs spanning multiple files or deep class hierarchies are truncated.
3. **Synthetic distribution**: The mutation strategies generate realistic patterns but may lack the nuanced mistakes seen in massive open-source commits.
"""))

nb['cells'] = cells

with open('c:/Users/Swayam/Desktop/NLP Assignment 2/bug-summarizer/notebook/train_codebert.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook written successfully.")
