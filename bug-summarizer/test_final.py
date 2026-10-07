import requests
import time
import subprocess

# Start the flask app
app_process = subprocess.Popen(
    ["c:/Users/Swayam/Desktop/NLP Assignment 2/venv/Scripts/python.exe", "app.py"],
    cwd="c:/Users/Swayam/Desktop/NLP Assignment 2/bug-summarizer"
)

# Wait for it to start
time.sleep(10)

url = "http://127.0.0.1:5000/predict"

snippets = [
    {
        "name": "add(a, b) missing return",
        "code": "def add(a, b):\n    result = a + b"
    },
    {
        "name": "square(x)",
        "code": "def square(x):\n    return x * x"
    },
    {
        "name": "Syntax Error",
        "code": "def bad()\n    print('hi')"
    },
    {
        "name": "Division by Zero",
        "code": "def divide():\n    return 1 / 0"
    },
    {
        "name": "Off by One",
        "code": "def off(arr):\n    for i in range(len(arr) + 1):\n        print(arr[i])"
    }
]

print("=== Testing Flask API (Final Fixes) ===")
for s in snippets:
    print(f"\nTesting: {s['name']}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json={"code": s["code"]})
            if response.status_code == 200:
                data = response.json()
                print(f"  Source: {data['source']}")
                print(f"  Prediction: {data['bug_type']}")
                if 'confidence' in data and data['source'] == 'codebert':
                    print(f"  Confidence: {data['confidence']}%")
                print(f"  Summary: {data['summary']}")
                if 'top3' in data and data['top3']:
                    top3_str = ", ".join([f"{x['label']} ({x['confidence']}%)" for x in data['top3']])
                    print(f"  Top 3: {top3_str}")
            else:
                print(f"  Error: HTTP {response.status_code} - {response.text}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            print(f"  Request failed: {e}")

# Kill the flask app
app_process.terminate()
