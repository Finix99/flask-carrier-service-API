from flask import Flask, request, jsonify
import joblib, json, pandas as pd, os
from datetime import datetime
from geopy.distance import geodesic
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

BASE_LOCATION = (-1.263757, 36.9116907)  # Shop base
MODEL_FILE = "xgb_delivery_model.json"
ENCODER_FILE = "city_encoder.pkl"
CONFIG_FILE = "config.json"
DATA_FILE = "delivery_history.csv"

API_KEY = os.getenv("FLASK_API_KEY", "b20f69591f2e4c906777e888437bb690")

# Load AI model & encoder if available
model, encoder = None, None
if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
    model = XGBRegressor()
    model.load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    print("✅ Loaded AI model and encoder.")
else:
    print("⚠️ AI model not found, using rule-based fallback.")

# Load fallback config
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        config = json.load(f)
else:
    config = {
        "rate_per_km_nairobi": 28,
        "base_fee_nairobi": 50,
        "flat_rate_others": 300,
        "free_shipping_minimum": 3000,
        "min_delivery_order": 500,
        "zone_surcharge": 10
    }

# Ensure delivery_history.csv exists
if not os.path.exists(DATA_FILE):
    pd.DataFrame(columns=[
        "timestamp","latitude","longitude","county",
        "distance_km","predicted_price_ksh","predicted_eta_hours"
    ]).to_csv(DATA_FILE, index=False)


@app.route("/")
def home():
    return jsonify({"message": "Nima Smart Shipping API active"})


@app.route("/predict-rate", methods=["POST"])
def predict_rate():
    client_key = request.headers.get("X-API-Key")
    if client_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    try:
        user_lat = float(data.get("latitude", 0))
        user_lon = float(data.get("longitude", 0))
        county = data.get("billing_address_2", "Unknown").strip()
        order_total = float(data.get("order_total", 0))
        timestamp = pd.to_datetime(data.get("timestamp", datetime.now().isoformat()))
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    distance_km = geodesic(BASE_LOCATION, (user_lat, user_lon)).km

    # Minimum order check
    if order_total < config["min_delivery_order"]:
        return jsonify({
            "error": f"Minimum order for delivery is KSh {config['min_delivery_order']}",
            "eligible": False
        }), 400

    # AI prediction if model exists
    if model and encoder:
        hour = timestamp.hour
        dayofweek = timestamp.dayofweek
        city_encoded = encoder.transform([county])[0] if county in encoder.classes_ else 0
        X_pred = pd.DataFrame([[distance_km, city_encoded, hour, dayofweek]],
                              columns=["distance_km", "city_encoded", "hour", "dayofweek"])
        predicted_price = model.predict(X_pred)[0]
        mode = "AI model"
    else:
        # Fallback rule-based
        if "nairobi" in county.lower():
            if order_total >= config["free_shipping_minimum"]:
                predicted_price = 0
            else:
                if distance_km <= 1.8:
                    predicted_price = config["base_fee_nairobi"]
                else:
                    extra_distance = distance_km - 1.8
                    predicted_price = config["base_fee_nairobi"] + (config["rate_per_km_nairobi"] * extra_distance)
        else:
            if order_total >= config["free_shipping_minimum"]:
                predicted_price = 0
            else:
                predicted_price = config["flat_rate_others"] + config["zone_surcharge"]
        mode = "rule-based"

    # Log prediction
    entry = {
        "timestamp": timestamp.isoformat(),
        "latitude": user_lat,
        "longitude": user_lon,
        "county": county,
        "distance_km": round(distance_km,2),
        "predicted_price_ksh": round(predicted_price,2),
        "predicted_eta_hours": None
    }
    df = pd.read_csv(DATA_FILE)
    df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)

    return jsonify({
        "distance_km": round(distance_km,2),
        "predicted_price_ksh": round(predicted_price,2),
        "mode": mode
    })


@app.route("/predict-eta", methods=["POST"])
def predict_eta():
    data = request.get_json()
    try:
        user_lat = float(data.get("latitude", 0))
        user_lon = float(data.get("longitude", 0))
        county = data.get("billing_address_2", "Unknown").strip()
        timestamp = pd.to_datetime(data.get("timestamp", datetime.now().isoformat()))
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    distance_km = geodesic(BASE_LOCATION, (user_lat, user_lon)).km

    # ETA logic
    if "nairobi" in county.lower():
        if distance_km <= 1.8:
            eta_hours = 0.5
        elif distance_km <= 5:
            eta_hours = 1.0
        else:
            eta_hours = 1.5
    else:
        eta_hours = 6.0

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

    return jsonify({
        "predicted_eta_hours": round(eta_hours,2),
        "eta_label": f"≈{int(eta_hours)}h",
        "mode": "rule-based"
    })


if __name__ == "__main__":
    app.run(debug=False)
