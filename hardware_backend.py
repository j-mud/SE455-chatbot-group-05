from flask import Flask, jsonify
from flask_cors import CORS
import serial
import time

app = Flask(__name__)
CORS(app)

PORT = "/dev/ttyUSB0"
BAUD = 9600

try:
    arduino = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)
except Exception as e:
    print(f"Warning: Could not connect to Arduino: {e}")
    arduino = None

latest_data = {
    "soil": "0",
    "temp": "0",
    "humidity": "0",
    "gas": "0",
    "status": "WAITING",
    "pump": "OFF",
    "ai_message": "Waiting for sensor data..."
}

def make_ai_message(data):
    status = data.get("status", "")
    if status == "HIGH_TEMP":
        return "High temperature detected. Heat stress risk. Activate irrigation."
    elif status == "HIGH_HUMIDITY":
        return "High humidity. Fungal disease risk. Activate fans."
    elif status == "GAS_ANOMALY":
        return "Gas level anomaly. Ventilation recommended."
    elif status == "COMBINED_STRESS":
        return "Multiple stress conditions. Immediate action required."
    elif status == "NORMAL":
        return "Greenhouse conditions are stable."
    else:
        return "Reading sensor data..."

@app.route("/sensor-data")
def sensor_data():
    global latest_data
    if arduino is None:
        return jsonify(latest_data)  
    line = arduino.readline().decode(errors="ignore").strip()

    if line:
        for part in line.split(","):
            if "=" in part:
                key, value = part.split("=")
                latest_data[key.strip()] = value.strip()

        latest_data["ai_message"] = make_ai_message(latest_data)

    return jsonify(latest_data)

if __name__ == "__main__":
    app.run(port=5000)