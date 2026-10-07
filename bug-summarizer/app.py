import os
import sys
import json
import re
import ast
import subprocess
import tempfile
import tokenize
import io
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# Configuration
# ============================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "codebert_bug_model")
MAX_LENGTH = 256
ML_CONFIDENCE_THRESHOLD = 0.70  # Only report ML finding as an issue above this

app = Flask(__name__)

# --- Load model at startup ---
print("Loading model and tokenizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    label_map_path = os.path.join(MODEL_DIR, "label_mapping.json")
    with open(label_map_path, "r") as f:
        label_mapping = json.load(f)
    id_to_label = label_mapping["id_to_label"]
    print(f"Model loaded successfully! {len(id_to_label)} classes.")
except Exception as e:
    print(f"WARNING: Could not load model: {e}")
    tokenizer = None
    model = None
    id_to_label = {}

# ============================================================
# Helpers
# ============================================================
BLOCK_HEADERS = re.compile(
    r"^(\s*)(def |class |if |elif |else|for |while |try|except|finally|with |async\s+def |async\s+for |async\s+with )"
)

def _clean_syntax_message(msg: str) -> str:
    """Remove raw parser artefacts like '(<unknown>, line 1)' and tidy up."""
    msg = re.sub(r"\(<unknown>,?\s*line\s*\d+\)", "", msg)
    msg = re.sub(r"\(<stdin>,?\s*line\s*\d+\)", "", msg)
    msg = re.sub(r"\(<string>,?\s*line\s*\d+\)", "", msg)
    msg = msg.strip().rstrip(".")
    # Capitalise first letter
    if msg:
        msg = msg[0].upper() + msg[1:]
    return msg

def _get_indentation(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]

def _is_block_header(line: str) -> bool:
    return bool(BLOCK_HEADERS.match(line))

def _repair_line(line: str, error_msg: str) -> str:
    """Try to repair a block header instead of masking it with 'pass'."""
    stripped = line.rstrip()
    # Missing colon at end of block header
    if _is_block_header(line) and not stripped.endswith(":"):
        # Check if the error is about the colon
        if "expected ':'" in error_msg.lower() or "invalid syntax" in error_msg.lower():
            return stripped + ":"
    return None  # Signal that we couldn't repair

def _mask_line(line: str) -> str:
    """Replace line content with 'pass', preserving original indentation."""
    indent = _get_indentation(line)
    return indent + "pass  # masked"


# ============================================================
# Step A1: Bracket / Quote Balance Check
# ============================================================
BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}"}
CLOSE_TO_OPEN = {v: k for k, v in BRACKET_PAIRS.items()}

def bracket_and_quote_check(code: str):
    """
    Walk through the source using Python's tokenizer when possible,
    otherwise fall back to a character-level scan that respects strings
    and comments. Reports each unclosed/mismatched bracket and each
    unterminated string with the line where it was opened.
    """
    errors = []
    stack = []  # (char, line_number)
    tokenizer_failed = False

    try:
        # Iterate over generator directly so tokens before any TokenError are processed
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            tok_type, tok_string, tok_start, _, _ = tok
            if tok_type == tokenize.OP:
                if tok_string in BRACKET_PAIRS:
                    stack.append((tok_string, tok_start[0]))
                elif tok_string in CLOSE_TO_OPEN:
                    expected_open = CLOSE_TO_OPEN[tok_string]
                    if stack and stack[-1][0] == expected_open:
                        stack.pop()
                    elif stack:
                        opener, open_line = stack.pop()
                        errors.append({
                            "type": "mismatched_bracket",
                            "line": open_line,
                            "message": (
                                f"'{opener}' opened on line {open_line} "
                                f"is closed with '{tok_string}' on line {tok_start[0]}"
                            ),
                            "source": "bracket_check",
                        })
                    else:
                        errors.append({
                            "type": "mismatched_bracket",
                            "line": tok_start[0],
                            "message": f"Unexpected closing '{tok_string}' with no matching opener",
                            "source": "bracket_check",
                        })
            elif tok_type == tokenize.ERRORTOKEN:
                # Unterminated string or other lexer error
                if tok_string in ("'", '"'):
                    errors.append({
                        "type": "unterminated_string",
                        "line": tok_start[0],
                        "message": f"Unterminated string starting on line {tok_start[0]}",
                        "source": "bracket_check",
                    })
    except tokenize.TokenError as e:
        msg = str(e)
        match = re.search(r"\((\d+),", msg)
        line = int(match.group(1)) if match else 1
        if "EOF in multi-line" in msg:
            # If stack is empty, it's an unclosed multi-line string/construct
            if not stack:
                errors.append({
                    "type": "unterminated_string",
                    "line": line,
                    "message": f"Unterminated string or expression starting on line {line}",
                    "source": "bracket_check",
                })
        elif "unterminated" in msg.lower():
            errors.append({
                "type": "unterminated_string",
                "line": line,
                "message": f"Unterminated string starting on line {line}",
                "source": "bracket_check",
            })
        else:
            tokenizer_failed = True
    except Exception:
        tokenizer_failed = True

    if tokenizer_failed:
        errors = _char_level_bracket_check(code)
    else:
        # Report any unclosed brackets left on the stack
        for opener, open_line in stack:
            errors.append({
                "type": "unclosed_bracket",
                "line": open_line,
                "message": f"'{opener}' opened on line {open_line} is never closed",
                "source": "bracket_check",
            })

    return errors


def _char_level_bracket_check(code: str):
    """Character-level bracket balance check that skips strings and comments."""
    errors = []
    stack = []
    lines = code.split("\n")
    in_string = None  # None, or the quote char(s)
    string_start_line = None

    for line_idx, line in enumerate(lines):
        i = 0
        while i < len(line):
            ch = line[i]

            # Skip comments (only outside strings)
            if in_string is None and ch == "#":
                break

            # String handling
            if in_string is not None:
                if line[i:i+len(in_string)] == in_string:
                    str_len = len(in_string)
                    in_string = None
                    i += str_len
                    continue
                elif ch == "\\":
                    i += 2
                    continue
                i += 1
                continue

            # Check for string start
            if ch in ('"', "'"):
                triple = line[i:i+3]
                if triple in ('"""', "'''"):
                    in_string = triple
                    string_start_line = line_idx + 1
                    i += 3
                else:
                    in_string = ch
                    string_start_line = line_idx + 1
                    i += 1
                continue

            # Brackets
            if ch in BRACKET_PAIRS:
                stack.append((ch, line_idx + 1))
            elif ch in CLOSE_TO_OPEN:
                expected = CLOSE_TO_OPEN[ch]
                if stack and stack[-1][0] == expected:
                    stack.pop()
                elif stack:
                    opener, open_line = stack.pop()
                    errors.append({
                        "type": "mismatched_bracket",
                        "line": open_line,
                        "message": (
                            f"'{opener}' opened on line {open_line} "
                            f"is closed with '{ch}' on line {line_idx + 1}"
                        ),
                        "source": "bracket_check",
                    })
                else:
                    errors.append({
                        "type": "mismatched_bracket",
                        "line": line_idx + 1,
                        "message": f"Unexpected closing '{ch}' with no matching opener",
                        "source": "bracket_check",
                    })
            i += 1

    # Check for unterminated string
    if in_string is not None:
        errors.append({
            "type": "unterminated_string",
            "line": string_start_line,
            "message": f"Unterminated string starting on line {string_start_line}",
            "source": "bracket_check",
        })

    # Check for unclosed brackets
    for opener, open_line in stack:
        errors.append({
            "type": "unclosed_bracket",
            "line": open_line,
            "message": f"'{opener}' opened on line {open_line} is never closed",
            "source": "bracket_check",
        })

    return errors


# ============================================================
# Step A2: Iterative AST Parsing (smart masking / repair)
# ============================================================
def iterative_ast_parse(code: str):
    """
    Parse code iteratively.  On each SyntaxError:
      - Record the error
      - If the offending line is a block header, try to repair it (e.g. add colon)
      - Otherwise mask the line with 'pass' preserving indentation
    Returns (errors_list, repaired_code_string).
    """
    errors = []
    lines = code.split("\n")
    repaired = list(lines)  # working copy
    seen_lines = set()
    max_iter = len(lines) + 5  # safety cap

    for _ in range(max_iter):
        try:
            ast.parse("\n".join(repaired))
            break  # clean parse
        except SyntaxError as e:
            lineno = getattr(e, "lineno", None)
            if lineno is None or lineno in seen_lines:
                # Can't make further progress
                msg = _clean_syntax_message(str(e))
                if msg:
                    errors.append({
                        "type": "syntax_error",
                        "line": lineno or 1,
                        "message": msg,
                        "source": "static_analysis",
                    })
                break

            seen_lines.add(lineno)
            raw_msg = str(e)
            clean_msg = _clean_syntax_message(raw_msg)

            err_type = "indentation_error" if "indent" in raw_msg.lower() else "syntax_error"
            errors.append({
                "type": err_type,
                "line": lineno,
                "message": clean_msg,
                "source": "static_analysis",
            })

            # Repair or mask
            idx = lineno - 1
            if 0 <= idx < len(repaired):
                repaired_line = _repair_line(repaired[idx], raw_msg)
                if repaired_line is not None:
                    repaired[idx] = repaired_line
                else:
                    repaired[idx] = _mask_line(repaired[idx])

    return errors, "\n".join(repaired)


# ============================================================
# Step A3: Pyflakes on repaired code
# ============================================================
def pyflakes_check(repaired_code: str):
    """Run pyflakes on the repaired copy.  Return (errors, warnings)."""
    errors = []
    warnings = []
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
        from io import StringIO

        err_stream = StringIO()
        warn_stream = StringIO()
        reporter = Reporter(warn_stream, err_stream)
        check(repaired_code, "<string>", reporter)

        # Pyflakes errors (undefined names etc.)
        for raw in err_stream.getvalue().strip().split("\n"):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(":")
            line_str = parts[1].strip() if len(parts) > 1 else "1"
            line = int(line_str) if line_str.isdigit() else 1
            msg = ":".join(parts[2:]).strip() if len(parts) > 2 else raw
            msg = _clean_syntax_message(msg)
            if "undefined name" in raw.lower():
                errors.append({
                    "type": "undefined_variable",
                    "line": line,
                    "message": msg,
                    "source": "pyflakes",
                })
            else:
                errors.append({
                    "type": "code_error",
                    "line": line,
                    "message": msg,
                    "source": "pyflakes",
                })

        # Pyflakes warnings (unused imports/variables)
        for raw in warn_stream.getvalue().strip().split("\n"):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(":")
            line_str = parts[1].strip() if len(parts) > 1 else "1"
            line = int(line_str) if line_str.isdigit() else 1
            msg = ":".join(parts[2:]).strip() if len(parts) > 2 else raw
            msg = _clean_syntax_message(msg)
            warnings.append({"line": line, "message": msg})

    except Exception:
        pass

    return errors, warnings


# ============================================================
# Step A4: Static AST rules on repaired code
# ============================================================
def ast_rule_checks(repaired_code: str):
    """Run static analysis rules on the repaired AST."""
    errors = []
    warnings = []
    try:
        tree = ast.parse(repaired_code)
    except SyntaxError:
        return errors, warnings

    for node in ast.walk(tree):
        # Rule: mutable default argument
        if isinstance(node, ast.FunctionDef):
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None and isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    errors.append({
                        "type": "mutable_default_argument",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' uses a mutable default argument",
                        "source": "ast_rules",
                    })

            # Rule: function with no return (and assigns a variable)
            has_return = False
            has_assignment = False
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value is not None:
                    has_return = True
                if isinstance(child, ast.Assign):
                    has_assignment = True
            if has_assignment and not has_return and node.name != "__init__":
                warnings.append({
                    "line": node.lineno,
                    "message": f"Function '{node.name}' assigns variables but has no return statement",
                })

    return errors, warnings


# ============================================================
# Step B: Safe Dynamic Check
# ============================================================
def dynamic_check(code: str):
    """
    Run ONLY the top-level statements the user actually wrote.
    Never auto-call functions with invented arguments.
    Skip if code only defines functions/classes or has syntax errors.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return [], []

    has_loop_or_recursion = False
    has_executable = False
    try:
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While, ast.AsyncFor, ast.FunctionDef, ast.AsyncFunctionDef)):
                has_loop_or_recursion = True

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef, ast.Import, ast.ImportFrom)):
                has_executable = True
                break

        if not has_executable:
            return [], []
    except Exception:
        return [], []

    temp_name = None
    errors = []
    notes = []
    try:
        temp_dir = tempfile.gettempdir()
        with tempfile.NamedTemporaryFile("w", suffix=".py", dir=temp_dir, delete=False, encoding="utf-8") as f:
            f.write(code)
            temp_name = f.name

        result = subprocess.run(
            [sys.executable, temp_name],
            capture_output=True,
            text=True,
            timeout=2.0,
            stdin=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            err_output = result.stderr.strip().split("\n")
            last_line = err_output[-1] if err_output else "Unknown runtime error"

            if "EOFError: EOF when reading a line" in last_line:
                notes.append({
                    "line": 1,
                    "message": "this program reads user input, so it was only checked statically and by CodeBERT"
                })
                return errors, notes

            line = 1
            for l in err_output:
                if "line " in l:
                    try:
                        line = int(l.split("line ")[1].split(",")[0])
                    except Exception:
                        pass

            err_type = "runtime_error"
            if "ZeroDivisionError" in last_line:
                err_type = "division_by_zero"
            elif "IndexError" in last_line:
                err_type = "list_index_out_of_bounds"
            elif "TypeError" in last_line:
                err_type = "type_error"
            elif "NameError" in last_line:
                err_type = "undefined_variable"
            elif "RecursionError" in last_line:
                err_type = "infinite_loop_risk"

            errors.append({
                "type": err_type,
                "line": line,
                "message": _clean_syntax_message(last_line),
                "source": "runtime_analysis",
            })
    except subprocess.TimeoutExpired:
        if has_loop_or_recursion:
            errors.append({
                "type": "infinite_loop_risk",
                "line": 1,
                "message": "Execution timed out — possible infinite loop or recursion",
                "source": "runtime_analysis",
            })
        else:
            notes.append({
                "line": 1,
                "message": "Execution timed out (no loops or recursion detected)",
            })
    except Exception:
        pass
    finally:
        if temp_name:
            try:
                if os.path.exists(temp_name):
                    os.remove(temp_name)
            except OSError:
                pass

    return errors, notes


# ============================================================
# Step C: CodeBERT Prediction
# ============================================================
def ml_predict(code: str):
    if model is None or tokenizer is None:
        return None

    inputs = tokenizer(
        code, max_length=MAX_LENGTH, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    top3_probs, top3_indices = torch.topk(probs, 3)

    top3 = []
    for p, idx in zip(top3_probs, top3_indices):
        label = id_to_label.get(str(idx.item()), "unknown")
        top3.append({"label": label, "confidence": p.item()})

    return top3


# ============================================================
# Deduplication
# ============================================================
def deduplicate_issues(issues: list) -> list:
    """
    Remove duplicates: if two detectors report the same problem on the same
    line, keep only one (prefer the more specific source).
    """
    seen = {}  # (normalised_type, line) -> issue
    SOURCE_PRIORITY = {
        "bracket_check": 0,
        "static_analysis": 1,
        "ast_rules": 2,
        "pyflakes": 3,
        "runtime_analysis": 4,
        "codebert": 5,
    }

    SYNTAX_TYPES = {
        "syntax_error", "indentation_error", "unclosed_bracket",
        "mismatched_bracket", "unterminated_string"
    }

    for issue in issues:
        raw_type = issue.get("type", "")
        # Normalise type for dedup — e.g. bracket errors and syntax errors
        # on the same line are the same defect; undefined_variable from pyflakes
        # and runtime are the same problem.
        if raw_type in SYNTAX_TYPES:
            norm_type = "syntax_group"
        else:
            norm_type = raw_type

        key = (norm_type, issue.get("line"))
        if key not in seen:
            seen[key] = issue
        else:
            # Keep the higher-priority (lower number) source
            existing_priority = SOURCE_PRIORITY.get(seen[key]["source"], 99)
            new_priority = SOURCE_PRIORITY.get(issue["source"], 99)
            if new_priority < existing_priority:
                seen[key] = issue

    def _sort_key(item):
        line = item.get("line")
        if isinstance(line, int):
            return (0, line)
        try:
            return (0, int(line))
        except (ValueError, TypeError):
            return (1, str(line))

    return sorted(seen.values(), key=_sort_key)


# ============================================================
# Summary Generation
# ============================================================
def generate_summary(issues: list, notes: list, ml_res: dict) -> str:
    """Generate a combined plain-English summary."""
    parts = []

    if issues:
        parts.append(f"Found {len(issues)} issue{'s' if len(issues) != 1 else ''}:")
        for i, issue in enumerate(issues, 1):
            bug_label = issue["type"].replace("_", " ").title()
            line_val = issue.get("line")
            line_str = f"[Line {line_val}] " if line_val != "N/A" and line_val is not None else ""
            parts.append(f"  {i}. {line_str}{bug_label} — {issue['message']}")

    if notes:
        parts.append("")
        parts.append(f"Notes ({len(notes)}):")
        for note in notes:
            parts.append(f"  • Line {note['line']}: {note['message']}")

    if ml_res:
        parts.append("")
        parts.append(ml_res["text"])

    if not issues and not notes and not ml_res:
        return "The code looks clean — no issues detected."

    if not issues and not ml_res:
        return "No errors detected. " + "\n".join(parts)

    return "\n".join(parts)


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data or "code" not in data:
            return jsonify({"error": "Missing code"}), 400

        code = data["code"].strip()
        if not code:
            return jsonify({"error": "Empty code"}), 400

        all_issues = []
        all_notes = []

        # ----------------------------------------------------------
        # Step A1: Bracket / quote balance
        # ----------------------------------------------------------
        bracket_errors = bracket_and_quote_check(code)
        all_issues.extend(bracket_errors)

        # ----------------------------------------------------------
        # Step A2: Iterative AST parse with smart repair/masking
        # ----------------------------------------------------------
        ast_errors, repaired_code = iterative_ast_parse(code)
        all_issues.extend(ast_errors)
        has_syntax_errors = (len(ast_errors) > 0 or len(bracket_errors) > 0)

        # ----------------------------------------------------------
        # Step A3: Pyflakes on repaired code
        # ----------------------------------------------------------
        pf_errors, pf_warnings = pyflakes_check(repaired_code)
        all_issues.extend(pf_errors)
        all_notes.extend(pf_warnings)  # warnings → notes, never block pipeline

        # ----------------------------------------------------------
        # Step A4: AST rules on repaired code
        # ----------------------------------------------------------
        rule_errors, rule_warnings = ast_rule_checks(repaired_code)
        all_issues.extend(rule_errors)
        all_notes.extend(rule_warnings)

        # ----------------------------------------------------------
        # Step B: Dynamic check (only if no syntax errors and has
        #         top-level executable code)
        # ----------------------------------------------------------
        if not has_syntax_errors:
            dyn_errors, dyn_notes = dynamic_check(code)
            all_issues.extend(dyn_errors)
            all_notes.extend(dyn_notes)

        # ----------------------------------------------------------
        # Step C: CodeBERT (only if no syntax errors)
        # ----------------------------------------------------------
        ml_result = None
        ml_top3 = []
        if not has_syntax_errors:
            raw_ml = ml_predict(code)
            if raw_ml:
                ml_top3 = raw_ml
                top = raw_ml[0]
                if top["label"] != "no_bug" and top["confidence"] >= ML_CONFIDENCE_THRESHOLD:
                    all_issues.append({
                        "type": top["label"],
                        "line": "N/A",
                        "message": f"CodeBERT detected a likely logical bug: {top['label'].replace('_', ' ')}",
                        "source": "codebert",
                    })
                    ml_result = {
                        "text": f"CodeBERT prediction: {top['label'].replace('_', ' ')} (high confidence).",
                    }
                elif top["label"] == "no_bug" and top["confidence"] >= 0.50:
                    ml_result = {
                        "text": "CodeBERT analysis: No issues found.",
                    }
                else:
                    # Uncertain — show top 3 without adding an issue
                    labels = ", ".join(
                        f"{p['label'].replace('_', ' ')}" for p in raw_ml[:3]
                    )
                    ml_result = {
                        "text": f"CodeBERT is uncertain. Top possibilities: {labels}. Please review manually.",
                    }

        # ----------------------------------------------------------
        # Deduplicate
        # ----------------------------------------------------------
        all_issues = deduplicate_issues(all_issues)

        # ----------------------------------------------------------
        # Build summary
        # ----------------------------------------------------------
        summary = generate_summary(all_issues, all_notes, ml_result)

        # ----------------------------------------------------------
        # Build response — no confidence percentages on tool findings
        # ----------------------------------------------------------
        response_issues = []
        for issue in all_issues:
            entry = {
                "type": issue["type"],
                "line": issue["line"],
                "message": issue["message"],
                "source": issue["source"],
            }
            response_issues.append(entry)

        response = {
            "issues": response_issues,
            "notes": all_notes,
            "summary": summary,
            "issue_count": len(response_issues),
        }

        # Include ML top-3 only when CodeBERT ran
        if ml_top3:
            response["ml_top3"] = [
                {"label": p["label"], "confidence": round(p["confidence"] * 100, 1)}
                for p in ml_top3
            ]

        return jsonify(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=5000)
