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

# --- Load trained model & label encoder ---
model = None
encoder = None
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    model = XGBRegressor()
    model.load_model(MODEL_FILE)
    import joblib
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
        "rate_per_km_nairobi": 150,
        "flat_rate_others": 450
    }

@app.route("/")
def home():
    return jsonify({"message": "Smart Shipping API active"})

@app.route("/predict-rate", methods=["POST"])
def predict_rate():
    data = request.get_json()

    try: 
        user_lat = float(data["latitude"])
        user_lon = float(data["longitude"])
        county = data.get("county", "Unknown")
        timestamp = data.get("timestamp", datetime.now().isoformat())
        timestamp = pd.to_datetime(timestamp)
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    # --- Compute distance ---
    distance_km = geodesic(BASE_LOCATION, (user_lat, user_lon)).km

    # --- If trained model exists ---
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
        return jsonify({
            "distance_km": round(distance_km, 2),
            "predicted_price_ksh": round(float(predicted_price), 2),
            "mode": "AI model"
        })
    else:
        # --- Fallback (no AI model yet) ---
        if "nairobi" in county.lower():
            price = config["rate_per_km_nairobi"] * distance_km
        else:
            price = config["flat_rate_others"]
        return jsonify({
            "distance_km": round(distance_km, 2),
            "predicted_price_ksh": round(price, 2),
            "mode": "fallback"
        })
if __name__ == "__main__":
    app.run(debug=False)  # debug=False is safer

