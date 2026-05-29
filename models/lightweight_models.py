
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import time
import psutil
from sklearn.utils.class_weight import compute_class_weight

LABEL_CLASSES = ["NORMAL", "HIGH_TEMP", "HIGH_HUMIDITY", "GAS_ANOMALY", "COMBINED_STRESS"]
FEATURES = ["temperature", "humidity", "gas_level"]


def load_data(csv_path="agriculture_dataset.csv"):
    df = pd.read_csv(csv_path)
    X = df[FEATURES].values
    le = LabelEncoder()
    le.fit(LABEL_CLASSES)
    y = le.transform(df["label"].values)
    return X, y, le


def train_all_models(X_train, y_train, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    models = {
        "DecisionTree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=20, max_depth=8, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42),
    }

    trained = {}
    for name, model in models.items():
        print(f"Training {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"  Done in {train_time:.2f}s")

        save_path = os.path.join(save_dir, f"{name}.pkl")
        joblib.dump(model, save_path)
        print(f"  Saved to {save_path}")
        trained[name] = model

    return trained


def evaluate_model(model, X_test, y_test, le, model_name="Model"):
    """Returns dict of all evaluation metrics."""
    process = psutil.Process(os.getpid())

    # Latency: measure per-sample inference time
    latencies = []
    for sample in X_test[:100]:  # first 100 samples for latency estimate
        start = time.perf_counter()
        model.predict(sample.reshape(1, -1))
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    # Full predictions
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    y_pred = model.predict(X_test)
    mem_after = process.memory_info().rss / 1024 / 1024

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    mem_usage   = mem_after - mem_before

    # Model file size
    model_path = f"saved_models/{model_name}.pkl"
    model_size_kb = os.path.getsize(model_path) / 1024 if os.path.exists(model_path) else 0

    results = {
        "model":          model_name,
        "accuracy":       round(acc, 4),
        "f1_weighted":    round(f1, 4),
        "avg_latency_ms": round(avg_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "memory_delta_mb": round(mem_usage, 2),
        "model_size_kb":  round(model_size_kb, 1),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  Accuracy:         {acc:.4f}")
    print(f"  F1 (weighted):    {f1:.4f}")
    print(f"  Avg Latency:      {avg_latency:.3f} ms")
    print(f"  P95 Latency:      {p95_latency:.3f} ms")
    print(f"  Memory Delta:     {mem_usage:.2f} MB")
    print(f"  Model Size:       {model_size_kb:.1f} KB")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=[0,1,2,3,4], target_names=le.classes_))
    return results


if __name__ == "__main__":
    print("Loading dataset...")
    X, y, le = load_data("agriculture_dataset.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    models = train_all_models(X_train, y_train)
    results = []
    for name, model in models.items():
        r = evaluate_model(model, X_test, y_test, le, name)
        results.append(r)

    df_results = pd.DataFrame(results)
    df_results.to_csv("results/lightweight_models_results.csv", index=False)
    print("\nResults saved to results/lightweight_models_results.csv")
