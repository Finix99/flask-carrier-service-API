from flask import Flask, request, jsonify
import pandas as pd, os, json
from datetime import datetime
from geopy.distance import geodesic
from xgboost import XGBRegressor
import joblib

app = Flask(__name__)

BASE_LOCATION = (-1.263757, 36.9116907)
DATA_FILE = "delivery_history.csv"
CONFIG_FILE = "config.json"
MODEL_FILE = "xgb_delivery_model.json"
ENCODER_FILE = "city_encoder.pkl"

API_KEY = os.getenv("FLASK_API_KEY", "b20f69591f2e4c906777e888437bb690")

# Load AI model and encoder
model, encoder = None, None
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    model = XGBRegressor()
    model.load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    print("✅ Loaded AI model and encoder.")
else:
    print("⚠️ AI model not found, fallback rules will be used.")

# Load config or use defaults
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        config = json.load(f)
else:
    config = {
        "base_fee_nairobi": 50,
        "rate_per_km_nairobi": 28,
        "flat_rate_others": 300,
        "zone_surcharge": 10,
        "free_shipping_minimum": 3000,
        "min_delivery_order": 500
    }

# Ensure CSV exists
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=[
        "timestamp","latitude","longitude","county",
        "distance_km","predicted_price_ksh","predicted_eta_hours"
    ]).to_csv(DATA_FILE,index=False)


@app.route("/predict-rate", methods=["POST"])
def predict_rate():
    # 🔒 API key check
    client_key = request.headers.get("X-API-Key")
    if client_key != API_KEY:
        return jsonify({"error":"Unauthorized"}), 401

    data = request.get_json()
    try:
        lat = float(data.get("latitude", 0))
        lon = float(data.get("longitude", 0))
        county = data.get("billing_address_2", "Unknown").strip()
        timestamp = pd.to_datetime(data.get("timestamp", datetime.now().isoformat()))
        order_total = float(data.get("order_total",0))
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    distance_km = geodesic(BASE_LOCATION,(lat, lon)).km

    # Minimum order check
    if order_total < config["min_delivery_order"]:
        return jsonify({
            "error": f"Minimum order for delivery is KSh {config['min_delivery_order']}",
            "eligible": False
        }), 400

    # Try AI prediction first
    if model and encoder:
        try:
            hour = timestamp.hour
            dayofweek = timestamp.dayofweek
            city_encoded = encoder.transform([county])[0] if county in encoder.classes_ else 0
            X_pred = pd.DataFrame([[distance_km, city_encoded, hour, dayofweek]],
                                  columns=["distance_km","city_encoded","hour","dayofweek"])
            predicted_price = float(model.predict(X_pred)[0])
            mode = "AI model"
        except Exception as e:
            predicted_price = None
            mode = "AI error"
    else:
        predicted_price = None
        mode = "No AI"

    # Fallback rule-based if AI not used or error
    if predicted_price is None:
        if "nairobi" in county.lower():
            if order_total >= config["free_shipping_minimum"]:
                predicted_price = 0
            else:
                if distance_km <= 1.8:
                    predicted_price = config["base_fee_nairobi"]
                else:
                    extra_km = distance_km - 1.8
                    predicted_price = config["base_fee_nairobi"] + (extra_km * config["rate_per_km_nairobi"])
            # ETA Nairobi
            if distance_km <= 1.8:
                eta_hours = 0.5
            elif distance_km <= 5:
                eta_hours = 1.0
            else:
                eta_hours = 1.5
        else:
            predicted_price = config["flat_rate_others"] + config["zone_surcharge"]
            eta_hours = 6.0
        mode = "rule-based"
    else:
        # If AI predicted, estimate ETA with same rule
        if "nairobi" in county.lower():
            if distance_km <= 1.8:
                eta_hours = 0.5
            elif distance_km <= 5:
                eta_hours = 1.0
            else:
                eta_hours = 1.5
        else:
            eta_hours = 6.0

    # Log prediction
    entry = {
        "timestamp": timestamp.isoformat(),
        "latitude": lat,
        "longitude": lon,
        "county": county,
        "distance_km": round(distance_km,2),
        "predicted_price_ksh": round(predicted_price,2),
        "predicted_eta_hours": eta_hours
    }
    df = pd.read_csv(DATA_FILE)
    df = pd.concat([df,pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DATA_FILE,index=False)

    return jsonify({
        "distance_km": round(distance_km,2),
        "predicted_price_ksh": round(predicted_price,2),
        "predicted_eta_hours": eta_hours,
        "mode": mode
    })


if __name__=="__main__":
    app.run(debug=False)
