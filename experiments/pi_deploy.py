import argparse
import time
import sys
import os
import joblib
import numpy as np
import psutil
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

LABEL_CLASSES = ["NORMAL", "HIGH_TEMP", "HIGH_HUMIDITY", "GAS_ANOMALY", "COMBINED_STRESS"]

INTENT_ACTIONS = {
    "NORMAL":           {"severity": "INFO",     "alert": False, "message": "All conditions nominal."},
    "HIGH_TEMP":        {"severity": "WARNING",  "alert": True,  "message": "Heat stress. Activate irrigation."},
    "HIGH_HUMIDITY":    {"severity": "WARNING",  "alert": True,  "message": "High humidity. Fungal risk."},
    "GAS_ANOMALY":      {"severity": "ALERT",    "alert": True,  "message": "Gas anomaly. Check ventilation."},
    "COMBINED_STRESS":  {"severity": "CRITICAL", "alert": True,  "message": "Multiple stressors. Immediate action."},
}


def load_model(model_path="saved_models/RandomForest.pkl"):
    model = joblib.load(model_path)
    le    = LabelEncoder()
    le.fit(LABEL_CLASSES)
    print(f"Model loaded: {model_path}")
    return model, le


def classify_reading(model, le, temperature, humidity, gas_level):
    features  = np.array([[temperature, humidity, gas_level]])
    start     = time.perf_counter()
    label_idx = model.predict(features)[0]
    latency   = (time.perf_counter() - start) * 1000
    label     = le.inverse_transform([label_idx])[0]
    proba     = model.predict_proba(features)[0]
    confidence = float(proba[label_idx])
    return label, confidence, latency


def log_decision(temperature, humidity, gas_level, label, confidence, latency):
    intent = INTENT_ACTIONS[label]
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line   = f"[{ts}] T={temperature}°C H={humidity}% G={gas_level} → {label} ({confidence*100:.1f}%) | {latency:.2f}ms"
    print(line)
    if intent["alert"]:
        print(f"  ⚠ {intent['severity']}: {intent['message']}")

    with open("results/pi_inference_log.csv", "a") as f:
        f.write(f"{ts},{temperature},{humidity},{gas_level},{label},{confidence:.4f},{latency:.3f}\n")


def run_serial(model, le, port="/dev/ttyUSB0", baud=9600):
    """Read from ESP32 over serial."""
    try:
        import serial
    except ImportError:
        print("pyserial not installed. Run: pip install pyserial")
        sys.exit(1)

    ser = serial.Serial(port, baud, timeout=2)
    print(f"Listening on {port} at {baud} baud...")

    while True:
        try:
            line = ser.readline().decode("utf-8").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue
            temperature = float(parts[0])
            humidity    = float(parts[1])
            gas_level   = float(parts[2])
            label, conf, lat = classify_reading(model, le, temperature, humidity, gas_level)
            log_decision(temperature, humidity, gas_level, label, conf, lat)
        except (ValueError, UnicodeDecodeError):
            pass
        except KeyboardInterrupt:
            print("\nStopped.")
            break


def run_simulate(model, le, n=30, interval=1.0):
    """Simulate sensor readings for testing without hardware."""
    import random

    scenarios = [
        (25.0, 65.0, 400),
        (41.5, 30.0, 380),
        (22.0, 92.0, 410),
        (27.0, 60.0, 850),
        (43.0, 91.0, 900),
    ]

    log_path = "results/pi_inference_log.csv"
    if not os.path.exists(log_path):
        os.makedirs("results", exist_ok=True)
        with open(log_path, "w") as f:
            f.write("timestamp,temperature,humidity,gas_level,label,confidence,latency_ms\n")

    print(f"Simulating {n} sensor readings (interval={interval}s)...\n")

    latencies = []
    process   = psutil.Process(os.getpid())

    for i in range(n):
        base_t, base_h, base_g = random.choice(scenarios)
        t = round(base_t + random.uniform(-1.5, 1.5), 1)
        h = round(base_h + random.uniform(-3.0, 3.0), 1)
        g = round(base_g + random.uniform(-20, 20), 0)

        mem = process.memory_info().rss / 1024 / 1024
        label, conf, lat = classify_reading(model, le, t, h, g)
        latencies.append(lat)

        log_decision(t, h, g, label, conf, lat)
        time.sleep(interval)

    print(f"\n{'='*50}")
    print(f"EDGE INFERENCE SUMMARY (on this device)")
    print(f"{'='*50}")
    print(f"  Total readings:  {n}")
    print(f"  Avg latency:     {np.mean(latencies):.3f} ms")
    print(f"  P95 latency:     {np.percentile(latencies, 95):.3f} ms")
    print(f"  Max latency:     {np.max(latencies):.3f} ms")
    print(f"  Current memory:  {process.memory_info().rss / 1024 / 1024:.1f} MB")
    print(f"\nLog saved to results/pi_inference_log.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["serial", "simulate"], default="simulate")
    parser.add_argument("--port",  default="/dev/ttyUSB0")
    parser.add_argument("--model", default="saved_models/RandomForest.pkl")
    parser.add_argument("--n",     type=int, default=30)
    args = parser.parse_args()

    model, le = load_model(args.model)

    if args.mode == "serial":
        run_serial(model, le, port=args.port)
    else:
        run_simulate(model, le, n=args.n)
