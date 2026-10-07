# 🐛 Python Source Code Bug Summarizer

**NLP Assignment — Semester 7**

A beginner-friendly Natural Language Processing project that uses a fine-tuned CodeBERT model to detect and summarize bugs in Python source code.

## 🌟 Project Overview
This project takes a snippet of Python code and uses an AI model to predict whether it contains a bug, what type of bug it is, and outputs a human-readable summary explaining the issue.

It uses:
- **Hugging Face Transformers** (`microsoft/codebert-base`)
- **PyTorch** for model training and inference
- **Flask** + HTML/CSS/JS for the web interface
- **Synthetic Data** generator to create 4,500 varied buggy and clean code examples

### Bug Classes Detected
1. `syntax_error` — Missing colons, brackets, or quotes.
2. `off_by_one` — Wrong loop range or index boundaries.
3. `wrong_operator` — Using `+` instead of `-`, `and` instead of `or`, etc.
4. `undefined_variable` — A variable is used before assignment.
5. `null_reference` — Operations on `None` values.
6. `missing_return` — A function computes a result but never returns it.
7. `wrong_comparison` — Flipped conditions, wrong logical boundaries.
8. `indentation_error` — Incorrect alignment of code blocks.
9. `no_bug` — Clean, correct code.

---

## 📂 Project Structure
```
bug-summarizer/
├── data/
│   └── bug_dataset.csv          # Synthetic dataset (~4,500 samples)
├── notebook/
│   └── train_codebert.ipynb     # Main training notebook for viva/presentation
├── model/
│   └── codebert_bug_model/      # Saved fine-tuned model (auto-created)
├── static/                      # Web app styles & JS
│   ├── style.css
│   └── script.js
├── templates/                   # Web app HTML
│   └── index.html
├── generate_dataset.py          # Script to create the synthetic dataset
├── train.py                     # Standalone training script
├── app.py                       # Flask web backend
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 How to Run

### 1. Setup Environment
Ensure you have Python 3.10+ installed.

```bash
# Create a virtual environment
python -m venv venv
venv\\Scripts\\activate   # On Windows

# Install dependencies (Install PyTorch CUDA version first if you have an Nvidia GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Generate the Dataset
Creates the `data/bug_dataset.csv` file.
```bash
python generate_dataset.py
```

### 3. Train the Model
You can train the model in two ways:
- **Interactive:** Open and run all cells in `notebook/train_codebert.ipynb` (best for learning/presentation).
- **Automated:** Run the training script directly:
```bash
python train.py
```
This will train CodeBERT and save the final model and tokenizer to `model/codebert_bug_model/`.

### 4. Run the Web App
Start the Flask application.
```bash
python app.py
```
Open your browser to `http://127.0.0.1:5000` to test the app!

---

## 🧠 How it Works (For Viva)

1. **Architecture:** CodeBERT is a Transformer encoder pre-trained on millions of lines of code. It understands code syntax and structure deeply.
2. **Fine-Tuning:** Since CodeBERT is encoder-only, we add a Sequence Classification head (a dense linear layer) on top. We train this head (and fine-tune the encoder) on our dataset to classify the input into one of 9 bug categories.
3. **Summary Generation:** Since it's a classification model and not a generative model (like GPT), we map the predicted bug class and the confidence score to pre-written, user-friendly summary templates. This is computationally efficient and prevents hallucinations.

---
*Built with ❤️ for NLP Assignment.*
