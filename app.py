from flask import Flask, request, jsonify
import joblib, json
import pandas as pd
from datetime import datetime
from geopy.distance import geodesic
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

BASE_LOCATION = (-1.263757, 36.9116907)  # Shop base
MODEL_FILE = "xgb_delivery_model.json"
ENCODER_FILE = "county_encoder.pkl"
CONFIG_FILE = "config.json"
DATA_FILE = "delivery_history.csv"

# --- Load trained model & label encoder ---
model = None
encoder = None
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    model = XGBRegressor()
    model.load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    print("✅ Loaded trained model and encoder.")
else:
    print("⚠️ No model found. Flask will use config-based fallback rates.")

# --- Load fallback config ---
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        config = json.load(f)
else:
    config = {
        "rate_per_km_nairobi": 90,
        "flat_rate_others": 300
    }
# --- Ensure delivery_history.csv exists ---
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=[
        "timestamp", "latitude", "longitude", "county",
        "distance_km", "predicted_price_ksh", "predicted_eta_hours"
    ]).to_csv(DATA_FILE, index=False)

from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# 🔒 API Key for authentication (set same key in WooCommerce plugin)
API_KEY = os.getenv("FLASK_API_KEY", "a94f2f3bd5f14a7f0a41ef73f6827c2e")

@app.route("/")
def home():
    return jsonify({"message": "Smart Shipping API active"})

@app.route("/predict-rate", methods=["POST"])
def predict_rate():
    # --- Authorization check ---
    client_key = request.headers.get("X-API-Key")
    if client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    # --- Parse request ---
    data = request.get_json()
    try:
        user_lat = float(data["latitude"])
        user_lon = float(data["longitude"])
        county = data.get("county", "Unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        timestamp = pd.to_datetime(timestamp)
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    distance_km = geodesic(BASE_LOCATION, (user_lat, user_lon)).km

    # --- AI prediction ---
    if model and encoder:
        hour = timestamp.hour
        dayofweek = timestamp.dayofweek
        if county not in encoder.classes_:
            county_encoded = 0
        else:
            county_encoded = encoder.transform([county])[0]
        X_pred = pd.DataFrame([[distance_km, county_encoded, hour, dayofweek]],
                              columns=["distance_km", "county_encoded", "hour", "dayofweek"])
        predicted_price = model.predict(X_pred)[0]
        mode = "Nima model"
    else:
        if "nairobi" in county.lower():
            predicted_price = config["rate_per_km_nairobi"] * distance_km
        else:
            predicted_price = config["flat_rate_others"]
        mode = "fallback"

    return jsonify({
        "distance_km": round(distance_km, 2),
        "predicted_price_ksh": round(float(predicted_price), 2),
        "mode": mode
    })

@app.route("/predict-eta", methods=["POST"])
def predict_eta():
    data = request.get_json()
    try:
        user_lat = float(data["latitude"])
        user_lon = float(data["longitude"])
        county = data.get("county", "Unknown")
        timestamp = pd.to_datetime(data.get("timestamp", datetime.now().isoformat()))
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    distance_km = geodesic(BASE_LOCATION, (user_lat, user_lon)).km
    eta_hours = 2.0 if "nairobi" in county.lower() else 6.0

    # Log prediction for future training
    entry = {
        "timestamp": timestamp.isoformat(),
        "latitude": user_lat,
        "longitude": user_lon,
        "county": county,
        "distance_km": round(distance_km,2),
        "predicted_price_ksh": None,
        "predicted_eta_hours": eta_hours
    }
    df = pd.read_csv(DATA_FILE)
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    return jsonify({"predicted_eta_hours": round(eta_hours,2),
                    "eta_label": f"≈{int(eta_hours)}h",
                    "mode":"fallback"})

@app.route("/log-delivery", methods=["POST"])
def log_delivery():
    data = request.get_json()
    try:
        order_ts = pd.to_datetime(data["order_timestamp"])
        delivered_ts = pd.to_datetime(data["delivered_timestamp"])
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        county = data.get("county", "Unknown")
        price = float(data.get("actual_price_ksh", 0))
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    distance_km = geodesic(BASE_LOCATION, (lat, lon)).km
    eta_hours = (delivered_ts - order_ts).total_seconds() / 3600.0

    df = pd.read_csv(DATA_FILE)
    entry = {
        "timestamp": order_ts.isoformat(),
        "latitude": lat,
        "longitude": lon,
        "county": county,
        "distance_km": round(distance_km,2),
        "predicted_price_ksh": price,
        "predicted_eta_hours": round(eta_hours,2)
    }
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    return jsonify({"status":"ok"}), 201

if __name__ == "__main__":
    app.run(debug=False)
