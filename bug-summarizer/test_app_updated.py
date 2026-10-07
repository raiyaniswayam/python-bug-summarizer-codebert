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
        "name": "square(x) (clean)",
        "code": "def square(x):\n    return x * x"
    },
    {
        "name": "Missing Return",
        "code": "def process_data(data):\n    result = data * 2"
    },
    {
        "name": "Undefined Variable",
        "code": "def calculate():\n    return non_existent_var + 10"
    },
    {
        "name": "Wrong Operator",
        "code": "def subtract(a, b):\n    return a + b"
    },
    {
        "name": "Off by One",
        "code": "def sum_elements(arr):\n    total = 0\n    for i in range(len(arr) + 1):\n        total += arr[i]\n    return total"
    }
]

print("=== Testing Flask API (Updated Dataset) ===")
for s in snippets:
    print(f"Testing: {s['name']}")
    try:
        response = requests.post(url, json={"code": s["code"]})
        if response.status_code == 200:
            data = response.json()
            print(f"  Prediction: {data['bug_type']} ({data['confidence']:.1f}%)")
            print(f"  Summary: {data['summary']}")
        else:
            print(f"  Error: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"  Request failed: {e}")
    print("-" * 30)

# Kill the flask app
app_process.terminate()
