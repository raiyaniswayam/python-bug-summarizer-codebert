# 🐛 Python Source Code Bug Summarizer — Build Plan

## Folder Layout
```
bug-summarizer/
├── data/
│   └── bug_dataset.csv          # ~4,500 synthetic samples
├── notebook/
│   └── train_codebert.ipynb     # Main training notebook
├── model/
│   └── codebert_bug_model/      # Saved fine-tuned model
├── app.py                       # Flask backend
├── templates/
│   └── index.html               # Frontend page
├── static/
│   ├── style.css                # Styles
│   └── script.js                # Client-side JS
├── generate_dataset.py          # Dataset generator script
├── requirements.txt             # Pinned dependencies
└── README.md                    # Full documentation
```

## Build Stages

| # | Stage | Status |
|---|-------|--------|
| 1 | Create folder structure & `requirements.txt` | ⏳ |
| 2 | Build `generate_dataset.py` and run it → `data/bug_dataset.csv` | ⏳ |
| 3 | Create Jupyter notebook (all 10 sections) | ⏳ |
| 4 | Run training, evaluate, save model | ⏳ |
| 5 | Build Flask app (`app.py`, templates, static) | ⏳ |
| 6 | Test web app with 5+ buggy snippets | ⏳ |
| 7 | Write `README.md` | ⏳ |
| 8 | Final summary & run commands | ⏳ |

## Bug Classes (9 total)
1. `syntax_error` — missing colons, brackets, quotes
2. `off_by_one` — wrong range bounds in loops
3. `wrong_operator` — using `+` instead of `-`, `=` instead of `==`, etc.
4. `undefined_variable` — using variables before assignment
5. `null_reference` — operations on `None`
6. `missing_return` — function computes but doesn't return
7. `wrong_comparison` — flipped conditions, wrong boolean logic
8. `indentation_error` — bad indentation in blocks
9. `no_bug` — clean, correct code

~500 samples per class → ~4,500 total, with 80/10/10 train/val/test split.
