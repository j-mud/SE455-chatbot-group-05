from flask import Flask, jsonify, request
from flask_cors import CORS
import serial
import time
import torch
import sqlite3
import os
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI

load_dotenv()  # reads OPENAI_API_KEY from .env automatically

app = Flask(__name__)
CORS(app)

SERIAL_PORT = "/dev/cu.usbmodemDCDA0C3CE6EC2"
BAUD_RATE = 9600

def connect_arduino():
    try:
        port = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"[Serial] Connected to {SERIAL_PORT}")
        return port
    except Exception as e:
        print(f"[Serial] Could not connect: {e}")
        return None

arduino = connect_arduino()

SENSOR_LABELS = [
    "NORMAL",
    "HIGH_TEMP",
    "HIGH_HUMIDITY",
    "GAS_ANOMALY",
    "COMBINED_STRESS",
]

COMMAND_LABELS = [
    "EXPLAIN_PUMP",
    "CHECK_SOIL",
    "CHECK_TEMP",
    "CHECK_HUMIDITY",
    "CHECK_GAS",
    "GENERAL_STATUS",
]

sensor_tokenizer = AutoTokenizer.from_pretrained("saved_models/distilbert", use_fast=False)
sensor_model = AutoModelForSequenceClassification.from_pretrained("saved_models/distilbert")
sensor_model.eval()

command_tokenizer = AutoTokenizer.from_pretrained("saved_models/command_slm", use_fast=False)
command_model = AutoModelForSequenceClassification.from_pretrained("saved_models/command_slm")
command_model.eval()

latest_data = {
    "soil": "0",
    "temp": "0",
    "humidity": "0",
    "gas": "0",
    "status": "WAITING",
    "pump": "OFF",
    "ai_message": "Waiting for real sensor readings.",
}

# ── Database ──────────────────────────────────────────────────────────────────

DB_PATH = "sensor_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            soil      TEXT,
            temp      TEXT,
            humidity  TEXT,
            gas       TEXT,
            pump      TEXT,
            status    TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

_api_key_loaded = bool(os.getenv("OPENAI_API_KEY"))
print(f"[OpenAI] {'API key loaded — AI responses enabled.' if _api_key_loaded else 'No API key found — falling back to rule-based responses.'}")

def save_reading(soil, temp, humidity, gas, pump, status):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sensor_readings (timestamp, soil, temp, humidity, gas, pump, status) VALUES (?,?,?,?,?,?,?)",
        (datetime.now().isoformat(), soil, temp, humidity, gas, pump, status)
    )
    conn.commit()
    conn.close()

def query_historical(minutes_ago_start, minutes_ago_end=None):
    """Fetch all rows in the requested time window (no LIMIT — we aggregate after)."""
    conn = sqlite3.connect(DB_PATH)
    older = (datetime.now() - timedelta(minutes=minutes_ago_start)).isoformat()
    if minutes_ago_end is not None:
        newer = (datetime.now() - timedelta(minutes=minutes_ago_end)).isoformat()
        rows = conn.execute(
            "SELECT timestamp, soil, temp, humidity, gas, pump, status FROM sensor_readings "
            "WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
            (older, newer)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT timestamp, soil, temp, humidity, gas, pump, status FROM sensor_readings "
            "WHERE timestamp >= ? ORDER BY timestamp ASC",
            (older,)
        ).fetchall()
    conn.close()
    return rows


def build_rag_context(question):
    """
    Retrieve relevant historical sensor data and format it as LLM context.
    Always includes the last 5 readings. Adds aggregated stats + samples
    when a time reference is detected in the question.
    """
    context = ""

    # Always attach the last 5 readings so the LLM has recent trend context
    recent = query_historical(2)
    if recent:
        context += "\nRecent readings (last 2 minutes):\n"
        for ts, s, t, h, g, p, _ in recent[-5:]:
            context += f"  [{ts[:19]}] Temp:{t}°C  Humidity:{h}%  Soil:{s}  Gas:{g}  Pump:{p}\n"

    # Check for a historical time reference
    time_ref = parse_time_reference(question)
    if not time_ref:
        return context

    start, end = time_ref
    rows = query_historical(start, end)

    if not rows:
        label = f"{start}–{end} min ago" if end else f"last {start} min"
        context += f"\nNo readings found in the database for {label}.\n"
        return context

    # Aggregate stats over the window
    def safe_floats(col):
        vals = []
        for r in rows:
            try:
                v = float(str(r[col]).rstrip('%'))
                if v > 0:
                    vals.append(v)
            except Exception:
                pass
        return vals

    temps  = safe_floats(2)
    hums   = safe_floats(3)
    soils  = safe_floats(1)
    gases  = safe_floats(4)

    label = f"around {start}–{end} min ago" if end else f"last {start} min"
    context += f"\nHistorical data ({len(rows)} readings, {label}):\n"

    if temps:
        context += f"  Temperature : avg {sum(temps)/len(temps):.1f}°C  min {min(temps):.1f}  max {max(temps):.1f}\n"
    if hums:
        context += f"  Humidity    : avg {sum(hums)/len(hums):.1f}%   min {min(hums):.1f}  max {max(hums):.1f}\n"
    if soils:
        context += f"  Soil Moisture: avg {sum(soils)/len(soils):.1f}%  min {min(soils):.1f}  max {max(soils):.1f}\n"
    if gases:
        context += f"  Gas Level   : avg {sum(gases)/len(gases):.0f}    min {min(gases):.0f}  max {max(gases):.0f}\n"

    # Evenly-spaced sample readings (5 points across the window)
    step = max(1, len(rows) // 5)
    samples = rows[::step][:5]
    context += "  Sample readings across window:\n"
    for ts, s, t, h, g, p, _ in samples:
        context += f"    [{ts[:19]}] Temp:{t}°C  Humidity:{h}%  Soil:{s}  Gas:{g}  Pump:{p}\n"

    return context

# ── Time reference parser (RAG retrieval key) ─────────────────────────────────

def parse_time_reference(question):
    """
    Returns (minutes_ago_start, minutes_ago_end) window to query, or None if no
    time reference is found. minutes_ago_end=None means 'up to now'.
    """
    q = question.lower()

    # Word-number map so "one minute ago", "two hours ago", etc. all work
    word_to_num = {
        'a': 1, 'an': 1, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'fifteen': 15, 'twenty': 20, 'thirty': 30, 'forty': 40, 'forty-five': 45,
        'sixty': 60,
    }

    # "X minutes ago" — digit or word
    m = re.search(r'(\d+|' + '|'.join(word_to_num) + r')\s*minutes?\s+ago', q)
    if m:
        raw = m.group(1)
        mins = int(raw) if raw.isdigit() else word_to_num[raw]
        window = max(3, mins // 3)          # ±window so we actually get rows
        return (mins + window, max(0, mins - window))

    # "half an hour ago" — must come before the generic hours pattern
    if re.search(r'half\s+(an|a)\s+hour', q):
        return (40, 20)

    # "X hours ago" — digit or word
    m = re.search(r'(\d+(?:\.\d+)?|' + '|'.join(word_to_num) + r')\s*hours?\s+ago', q)
    if m:
        raw = m.group(1)
        mins = int(float(raw) * 60) if re.match(r'[\d.]+', raw) else word_to_num[raw] * 60
        window = max(10, mins // 6)
        return (mins + window, max(0, mins - window))

    # "last / past X minutes"
    m = re.search(r'(last|past)\s+(\d+|' + '|'.join(word_to_num) + r')\s*minutes?', q)
    if m:
        raw = m.group(2)
        mins = int(raw) if raw.isdigit() else word_to_num[raw]
        return (mins, None)

    # "last / past X hours"
    m = re.search(r'(last|past)\s+(\d+|' + '|'.join(word_to_num) + r')\s*hours?', q)
    if m:
        raw = m.group(2)
        hrs = int(raw) if raw.isdigit() else word_to_num[raw]
        return (hrs * 60, None)

    # "last / past hour"
    if re.search(r'(past|last)\s+hour', q):
        return (60, None)

    # "today"
    if 'today' in q:
        now = datetime.now()
        return (now.hour * 60 + now.minute, None)

    # "a moment ago", "just now", "recently", "earlier", "a while ago"
    if re.search(r'(just now|a moment ago|moments? ago)', q):
        return (3, None)

    if any(w in q for w in ['earlier', 'recently', 'a while ago', 'before']):
        return (30, None)

    return None

# ── AI response (OpenAI + RAG) ────────────────────────────────────────────────

def get_ai_response(question):
    soil     = latest_data["soil"]
    temp     = latest_data["temp"]
    humidity = latest_data["humidity"]
    gas      = latest_data["gas"]
    pump     = latest_data["pump"]
    status   = latest_data["status"]
    ai_msg   = latest_data["ai_message"]

    # RAG: retrieve historical context from local SQLite database
    historical_context = build_rag_context(question)

    system_prompt = (
        "You are a smart greenhouse monitoring assistant. Help the farmer understand "
        "their greenhouse conditions and make good irrigation decisions. Be concise, "
        "natural, and accurate.\n\n"
        "Current live sensor readings:\n"
        f"  Temperature : {temp}°C\n"
        f"  Humidity    : {humidity}%\n"
        f"  Soil Moisture: {soil} (0%=dry, 100%=saturated; pump on below {SOIL_PUMP_THRESHOLD_PCT}%)\n"
        f"  Gas Level   : {gas}\n"
        f"  Pump Status : {pump}\n"
        f"  System Status: {status}\n"
        f"  Assessment  : {ai_msg}"
        f"{historical_context}\n\n"
        "Use the sensor data above to answer the user's question. "
        "If historical data is provided, use it to answer time-based questions accurately."
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Graceful fallback when no key is configured
        return answer_command(predict_command_intent(question))

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return answer_command(predict_command_intent(question))

# ── SLM helpers (unchanged, kept as fallback) ─────────────────────────────────

# Calibration constants for ESP32 12-bit ADC capacitive soil sensor.
# DRY_ADC  = reading in open air (no soil contact) — adjust to match your sensor.
# WET_ADC  = reading submerged / fully saturated   — adjust to match your sensor.
SOIL_DRY_ADC = 4095   # sensor reading in open air (confirmed)
SOIL_WET_ADC = 3414   # sensor reading submerged in water (back-calculated from 22% reading)

# Pump turns on when moisture drops below this percentage
SOIL_PUMP_THRESHOLD_PCT = 30  # pump on below 30% — water only reaches ~22% so threshold must be above dry baseline

def normalize_soil(raw_adc):
    """Convert raw ESP32 ADC value to 0–100% moisture (100 = saturated)."""
    raw = float(raw_adc)
    pct = (SOIL_DRY_ADC - raw) / (SOIL_DRY_ADC - SOIL_WET_ADC) * 100
    return round(max(0.0, min(100.0, pct)), 1)

def predict_sensor_status(temp, humidity, gas):
    # Text format must match training data: "temperature X humidity Y gas Z"
    text = f"temperature {temp} humidity {humidity} gas {gas}"
    inputs = sensor_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=32)
    with torch.no_grad():
        pred_id = torch.argmax(sensor_model(**inputs).logits, dim=1).item()
    return SENSOR_LABELS[pred_id]


def predict_command_intent(question):
    q = question.lower()
    if "pump" in q or "water" in q or "irrigation" in q:
        return "EXPLAIN_PUMP"
    inputs = command_tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=32)
    with torch.no_grad():
        pred_id = torch.argmax(command_model(**inputs).logits, dim=1).item()
    return COMMAND_LABELS[pred_id]


def make_sensor_message(status):
    messages = {
        "COMBINED_STRESS": "Multiple risk conditions detected. Irrigation and monitoring are recommended.",
        "HIGH_TEMP":       "High temperature detected. Monitor greenhouse conditions.",
        "HIGH_HUMIDITY":   "Humidity is above the safe range. Irrigation is not recommended.",
        "GAS_ANOMALY":     "Gas level is above normal range. Check air quality.",
        "NORMAL":          "Environmental conditions are stable.",
    }
    return messages.get(status, "Waiting for valid sensor readings.")


def answer_command(intent):
    soil     = latest_data["soil"]
    temp     = latest_data["temp"]
    humidity = latest_data["humidity"]
    gas      = latest_data["gas"]
    pump     = latest_data["pump"]
    status   = latest_data["status"]

    if intent == "EXPLAIN_PUMP":
        if pump == "ON":
            return f"The pump is ON because the soil moisture reading is {soil}, so irrigation is required."
        return f"The pump is OFF because the soil moisture reading is {soil}, so irrigation is not required."
    if intent == "CHECK_SOIL":
        return f"Current soil moisture is {soil}. Condition classified as {status}."
    if intent == "CHECK_TEMP":
        return f"Current temperature is {temp}°C. Condition classified as {status}."
    if intent == "CHECK_HUMIDITY":
        return f"Current humidity is {humidity}%. Condition classified as {status}."
    if intent == "CHECK_GAS":
        return f"Current gas reading is {gas}. Condition classified as {status}."
    return f"Current condition: {status}. {latest_data['ai_message']}"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/sensor-data")
def sensor_data():
    global latest_data
    global arduino
    if arduino is None:
        arduino = connect_arduino()
        if arduino is None:
            return jsonify(latest_data)

    try:
        line = arduino.readline().decode("utf-8").strip()
        if line and "=" in line:
            print("Arduino:", line)
            parsed = {}
            for part in line.split(","):
                if "=" in part:
                    key, value = part.split("=", 1)
                    parsed[key.strip()] = value.strip()

            soil_raw = parsed.get("soil",     "0")
            gas      = parsed.get("gas",      "0")
            pump     = parsed.get("pump",     "OFF")   # trust Arduino relay logic

            # Keep last valid temp/humidity if Arduino reports nan (DHT error)
            raw_temp = parsed.get("temp", "0")
            raw_hum  = parsed.get("humidity", "0")
            temp     = raw_temp if raw_temp.lower() not in ("nan", "") else latest_data["temp"]
            humidity = raw_hum  if raw_hum.lower()  not in ("nan", "") else latest_data["humidity"]

            soil_pct      = normalize_soil(soil_raw)
            sensor_status = predict_sensor_status(temp, humidity, gas)
            latest_data = {
                "soil":       f"{soil_pct}%",
                "temp":       temp,
                "humidity":   humidity,
                "gas":        gas,
                "status":     sensor_status,
                "pump":       pump,
                "ai_message": make_sensor_message(sensor_status),
            }

            save_reading(f"{soil_pct}%", temp, humidity, gas, pump, sensor_status)

    except Exception as e:
        print("Serial read error:", e)
        try:
            arduino.close()
        except Exception:
            pass
        arduino = None   # will reconnect on next request

    return jsonify(latest_data)


@app.route("/ask-command", methods=["POST"])
def ask_command():
    question = request.json.get("question", "")
    answer = get_ai_response(question)
    return jsonify({"answer": answer})


@app.route("/pump-control", methods=["POST"])
def pump_control():
    """Send a manual pump command to the Arduino: ON, OFF, or AUTO."""
    state = request.json.get("state", "AUTO").upper()
    if state not in ("ON", "OFF", "AUTO"):
        return jsonify({"error": "state must be ON, OFF, or AUTO"}), 400
    try:
        arduino.write(f"PUMP:{state}\n".encode())
        return jsonify({"message": f"Pump command sent: {state}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
