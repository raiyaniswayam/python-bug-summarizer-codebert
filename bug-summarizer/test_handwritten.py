import json
import os
import requests
import time
import subprocess
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Starting Flask server for testing...")
    app_process = subprocess.Popen(
        ["c:/Users/Swayam/Desktop/NLP Assignment 2/venv/Scripts/python.exe", "app.py"],
        cwd="c:/Users/Swayam/Desktop/NLP Assignment 2/bug-summarizer"
    )
    time.sleep(10)
    
    with open("data/handwritten_test.json", "r") as f:
        data = json.load(f)
        
    url = "http://127.0.0.1:5000/predict"
    y_true = []
    y_pred = []
    
    print(f"Testing {len(data)} handwritten snippets through the Hybrid Pipeline...")
    
    correct = 0
    for idx, item in enumerate(data):
        code = item["code"]
        label = item["label"]
        
        try:
            resp = requests.post(url, json={"code": code})
            if resp.status_code == 200:
                res_data = resp.json()
                pred = res_data["bug_type"]
                source = res_data.get("source", "unknown")
                y_true.append(label)
                y_pred.append(pred)
                
                is_correct = (pred == label)
                if is_correct:
                    correct += 1
                
                print(f"[{idx+1}/{len(data)}] True: {label:25s} | Pred: {pred:25s} | Source: {source:15s} | {'✅' if is_correct else '❌'}")
            else:
                print(f"[{idx+1}/{len(data)}] Error {resp.status_code}")
                y_true.append(label)
                y_pred.append("error")
        except Exception as e:
            print(f"[{idx+1}/{len(data)}] Request failed: {e}")
            y_true.append(label)
            y_pred.append("error")
            
    print("-" * 60)
    print(f"Hybrid Pipeline Handwritten Accuracy: {correct}/{len(data)} ({(correct/len(data))*100:.1f}%)")
    
    # We can only report metrics for classes that appear in y_true and y_pred
    unique_labels = list(set(y_true))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=unique_labels, zero_division=0))
    
    app_process.terminate()

if __name__ == "__main__":
    main()
