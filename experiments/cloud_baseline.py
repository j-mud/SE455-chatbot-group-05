import sys
import time
import numpy as np
import pandas as pd
import joblib
import requests
import os


def run_server(host="0.0.0.0", port=5001):
    from flask import Flask, request, jsonify
    app = Flask(__name__)

    model = joblib.load("saved_models/RandomForest.pkl")

    LABEL_CLASSES = ["NORMAL", "HIGH_TEMP", "HIGH_HUMIDITY", "GAS_ANOMALY", "COMBINED_STRESS"]
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(LABEL_CLASSES)

    @app.route("/predict", methods=["POST"])
    def predict():
        time.sleep(0.05)
        data = request.json
        features = np.array([[data["temperature"], data["humidity"], data["gas_level"]]])
        label_idx = model.predict(features)[0]
        label = le.inverse_transform([label_idx])[0]
        proba = model.predict_proba(features)[0]
        confidence = float(proba[label_idx])
        return jsonify({"prediction": label, "confidence": round(confidence, 4)})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    print(f"Cloud baseline server running at http://{host}:{port}")
    app.run(host=host, port=port)


def run_client(host="127.0.0.1", port=5001, n_requests=50):
    url = f"http://{host}:{port}/predict"

    test_samples = [
        {"temperature": 25.0, "humidity": 65.0, "gas_level": 400},
        {"temperature": 41.5, "humidity": 30.0, "gas_level": 380},
        {"temperature": 22.0, "humidity": 92.0, "gas_level": 410},
        {"temperature": 27.0, "humidity": 60.0, "gas_level": 850},
        {"temperature": 43.0, "humidity": 91.0, "gas_level": 900},
    ]

    print(f"Sending {n_requests} requests to {url}...")
    latencies = []
    errors = 0

    for i in range(n_requests):
        sample = test_samples[i % len(test_samples)]
        try:
            start = time.perf_counter()
            resp  = requests.post(url, json=sample, timeout=5)
            lat   = (time.perf_counter() - start) * 1000
            latencies.append(lat)
            if (i + 1) % 10 == 0:
                print(f"  Request {i+1}/{n_requests} | Latency: {lat:.1f} ms")
        except Exception as e:
            errors += 1
            print(f"  Request {i+1} failed: {e}")

    if latencies:
        results = {
            "approach":       "Cloud (simulated)",
            "n_requests":     n_requests,
            "errors":         errors,
            "avg_latency_ms": round(np.mean(latencies), 2),
            "min_latency_ms": round(np.min(latencies), 2),
            "max_latency_ms": round(np.max(latencies), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
            "p99_latency_ms": round(np.percentile(latencies, 99), 2),
        }

        print("\n" + "="*50)
        print("CLOUD BASELINE LATENCY RESULTS")
        print("="*50)
        for k, v in results.items():
            print(f"  {k}: {v}")

        os.makedirs("results", exist_ok=True)
        pd.DataFrame([results]).to_csv("results/cloud_baseline_latency.csv", index=False)
        print("\nSaved to results/cloud_baseline_latency.csv")
    else:
        print("No successful requests. Check server connection.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python experiments/cloud_baseline.py server")
        print("  python experiments/cloud_baseline.py client --host <ip>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "server":
        run_server()
    elif mode == "client":
        host = "127.0.0.1"
        if "--host" in sys.argv:
            idx = sys.argv.index("--host")
            host = sys.argv[idx + 1]
        run_client(host=host)
    else:
        print(f"Unknown mode: {mode}")
