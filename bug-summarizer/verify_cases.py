import requests
import json
import sys
import time

URL = "http://127.0.0.1:5000/predict"

def check_liveness():
    try:
        r = requests.get("http://127.0.0.1:5000/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# (a) Clean 30-line program
clean_code = """def add(a, b):
    \"\"\"Add two numbers\"\"\"
    return a + b

def subtract(a, b):
    \"\"\"Subtract b from a\"\"\"
    return a - b

def multiply(a, b):
    \"\"\"Multiply two numbers\"\"\"
    return a * b

def divide_safe(a, b):
    \"\"\"Safe division\"\"\"
    if b == 0:
        return 0
    return a / b

def square(x):
    \"\"\"Calculate square\"\"\"
    return x * x

def cube(x):
    \"\"\"Calculate cube\"\"\"
    return x * x * x

def is_positive(x):
    \"\"\"Check positivity\"\"\"
    return x > 0

def identity(x):
    \"\"\"Return value unchanged\"\"\"
    return x
"""

# (b) Function with missing colon and unclosed bracket
broken_colon_bracket_code = """def calculate(items)
    result = sum([x * 2 for x in items
    return result"""

# (c) Infinite loop snippet
infinite_loop_code = """count = 0
while True:
    count += 1
"""

# (d) Snippet with division by zero
div_zero_code = """def compute():
    numerator = 100
    denominator = 0
    return numerator / denominator

compute()
"""

tests = [
    {
        "id": "(a)",
        "name": "Clean 30-line program",
        "code": clean_code,
        "line_count": len(clean_code.strip().split("\n")),
        "expected": "0 issues",
    },
    {
        "id": "(b)",
        "name": "Function with missing colon and unclosed bracket",
        "code": broken_colon_bracket_code,
        "line_count": len(broken_colon_bracket_code.strip().split("\n")),
        "expected": "2 issues",
    },
    {
        "id": "(c)",
        "name": "Infinite loop snippet",
        "code": infinite_loop_code,
        "line_count": len(infinite_loop_code.strip().split("\n")),
        "expected": "Timeout message and server stays alive",
    },
    {
        "id": "(d)",
        "name": "Snippet with division by zero",
        "code": div_zero_code,
        "line_count": len(div_zero_code.strip().split("\n")),
        "expected": "Division by zero reported and server stays alive",
    },
]

print("=" * 60)
print("RUNNING AUTOMATED VERIFICATION SUITE")
print("=" * 60)

all_passed = True

for t in tests:
    print(f"\n--- Test {t['id']}: {t['name']} ---")
    start = time.time()
    try:
        r = requests.post(URL, json={"code": t["code"]}, timeout=10)
        elapsed = time.time() - start
        print(f"HTTP Status: {r.status_code} (took {elapsed:.2f}s)")
        
        if r.status_code == 200:
            data = r.json()
            issues = data.get("issues", [])
            issue_count = data.get("issue_count", len(issues))
            print(f"Issue count: {issue_count}")
            for idx, issue in enumerate(issues, 1):
                print(f"  Issue {idx}: [Line {issue.get('line')}] {issue.get('type')} ({issue.get('source')}) - {issue.get('message')}")
            
            # Validation
            if t["id"] == "(a)":
                if issue_count == 0:
                    print("  >>> PASS: No issues found for clean code.")
                else:
                    print("  >>> FAIL: Expected 0 issues.")
                    all_passed = False
            elif t["id"] == "(b)":
                if issue_count == 2:
                    print("  >>> PASS: Exactly 2 issues found.")
                else:
                    print(f"  >>> FAIL: Expected 2 issues, got {issue_count}.")
                    all_passed = False
            elif t["id"] == "(c)":
                has_timeout = any("timeout" in i.get("message", "").lower() or "infinite_loop" in i.get("type", "").lower() for i in issues)
                if has_timeout:
                    print("  >>> PASS: Timeout reported.")
                else:
                    print("  >>> FAIL: Timeout not reported.")
                    all_passed = False
            elif t["id"] == "(d)":
                has_div_zero = any("division_by_zero" in i.get("type", "") or "zero" in i.get("message", "").lower() for i in issues)
                if has_div_zero:
                    print("  >>> PASS: Division by zero reported.")
                else:
                    print("  >>> FAIL: Division by zero not reported.")
                    all_passed = False
        else:
            print(f"Server returned error: {r.text}")
            all_passed = False

    except Exception as e:
        print(f"Request failed: {e}")
        all_passed = False

    alive = check_liveness()
    print(f"Server alive check after test: {'ALIVE' if alive else 'DEAD!'}")
    if not alive:
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("ALL TESTS PASSED SUCCESSFULLY!")
else:
    print("SOME TESTS FAILED.")
print("=" * 60)
