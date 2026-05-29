import joblib
import numpy as np
from datetime import datetime

LABEL_CLASSES = ["NORMAL", "HIGH_TEMP", "HIGH_HUMIDITY", "GAS_ANOMALY", "COMBINED_STRESS"]


INTENT_ACTIONS = {
    "NORMAL": {
        "severity":  "INFO",
        "alert":     False,
        "message":   "All conditions nominal. No action required.",
        "commands":  []
    },
    "HIGH_TEMP": {
        "severity":  "WARNING",
        "alert":     True,
        "message":   "Heat stress detected. Crop damage risk above 35°C.",
        "commands":  [
            "ACTIVATE: Irrigation system",
            "OPEN: Greenhouse ventilation vents",
            "NOTIFY: Check crop shading"
        ]
    },
    "HIGH_HUMIDITY": {
        "severity":  "WARNING",
        "alert":     True,
        "message":   "High humidity detected. Fungal disease risk elevated.",
        "commands":  [
            "ACTIVATE: Circulation fans",
            "SCHEDULE: Fungicide application check",
            "REDUCE: Irrigation frequency"
        ]
    },
    "GAS_ANOMALY": {
        "severity":  "ALERT",
        "alert":     True,
        "message":   "Gas level anomaly detected. Possible NH3 or CO2 buildup.",
        "commands":  [
            "ACTIVATE: Emergency ventilation",
            "CHECK: Soil drainage and fertilizer levels",
            "NOTIFY: Manual sensor inspection required"
        ]
    },
    "COMBINED_STRESS": {
        "severity":  "CRITICAL",
        "alert":     True,
        "message":   "Multiple stress conditions detected simultaneously. Immediate action required.",
        "commands":  [
            "ACTIVATE: All ventilation and cooling systems",
            "SUSPEND: Fertilizer application",
            "NOTIFY: Immediate manual intervention required",
            "LOG: Critical event for review"
        ]
    }
}


class IntentProcessor:

    def __init__(self, model_path="saved_models/RandomForest.pkl", label_encoder=None):
        self.model = joblib.load(model_path)
        self.label_encoder = label_encoder
        self.log = []

    def process(self, temperature: float, humidity: float, gas_level: float) -> dict:
        features = np.array([[temperature, humidity, gas_level]])

        
        label_idx = self.model.predict(features)[0]
        if self.label_encoder:
            label = self.label_encoder.inverse_transform([label_idx])[0]
        else:
            label = LABEL_CLASSES[label_idx]

        
        proba = self.model.predict_proba(features)[0]
        confidence = round(float(proba[label_idx]), 4)

        
        intent = INTENT_ACTIONS[label]

        result = {
            "timestamp":   datetime.now().isoformat(),
            "inputs": {
                "temperature": temperature,
                "humidity":    humidity,
                "gas_level":   gas_level
            },
            "prediction":  label,
            "confidence":  confidence,
            "severity":    intent["severity"],
            "alert":       intent["alert"],
            "message":     intent["message"],
            "commands":    intent["commands"]
        }

        self._log_event(result)
        return result

    def _log_event(self, result: dict):
        self.log.append(result)
        status = f"[{result['severity']}] {result['timestamp']} | {result['prediction']} ({result['confidence']*100:.1f}%)"
        print(status)
        if result["alert"]:
            print(f"  → {result['message']}")
            for cmd in result["commands"]:
                print(f"     • {cmd}")

    def get_log(self):
        return self.log



def simulate_sensor_feed(processor: IntentProcessor, n_readings=10):
    import random
    import time

    scenarios = [
        (25.0, 65.0, 400),   # normal
        (41.5, 30.0, 380),   # high t
        (22.0, 92.0, 410),   # high h
        (27.0, 60.0, 850),   # gas
        (43.0, 91.0, 900),   # stress
    ]

    print("\n" + "="*60)
    print("SIMULATED SENSOR FEED")
    print("="*60)

    for i in range(n_readings):
        temp, hum, gas = random.choice(scenarios)
        # Add small noise
        temp += random.uniform(-1.5, 1.5)
        hum  += random.uniform(-3.0, 3.0)
        gas  += random.uniform(-20, 20)

        result = processor.process(
            temperature=round(temp, 1),
            humidity=round(hum, 1),
            gas_level=round(gas, 0)
        )
        time.sleep(0.1)  

    print("\nDone. Total events logged:", len(processor.log))
    alerts = sum(1 for r in processor.log if r["alert"])
    print(f"Alerts triggered: {alerts}/{len(processor.log)}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")

    # Load label encoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(LABEL_CLASSES)

    processor = IntentProcessor(
        model_path="saved_models/RandomForest.pkl",
        label_encoder=le
    )

    simulate_sensor_feed(processor, n_readings=15)
