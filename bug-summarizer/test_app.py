import requests
import time
import subprocess
import os

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
        "name": "Missing Return",
        "code": "def process_data(data):\n    result = data * 2\n    # No return"
    },
    {
        "name": "Undefined Variable",
        "code": "def calculate():\n    return x + 10"
    },
    {
        "name": "Off by One",
        "code": "def sum_list(lst):\n    total = 0\n    for i in range(len(lst) + 1):\n        total += lst[i]\n    return total"
    },
    {
        "name": "Wrong Operator",
        "code": "def get_average(scores):\n    total = sum(scores)\n    return total * len(scores)"
    },
    {
        "name": "Clean Code",
        "code": "def add(a, b):\n    return a + b"
    }
]

print("=== Testing Flask API ===")
for s in snippets:
    print(f"Testing: {s['name']}")
    try:
        response = requests.post(url, json={"code": s["code"]})
        if response.status_code == 200:
            data = response.json()
            print(f"  Prediction: {data['bug_type']} ({data['confidence']}%)")
            print(f"  Summary: {data['summary']}")
        else:
            print(f"  Error: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Request failed: {e}")
    print("-" * 30)

# Kill the flask app
app_process.terminate()
