import os
import json
import joblib
import pickle
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

# Database config
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# Load models and data
model_A = joblib.load("model_compA.pkl")
model_B = joblib.load("model_compB.pkl")
prices = pd.read_csv("data/product_prices_leaflets.csv")

# Preprocess price data
prices = prices[prices["discount"] >= 0].copy()
prices["final_price"] = prices["pvp_was"] * (1 - prices["discount"])
prices["date"] = pd.to_datetime(prices["time_key"].astype(str), format="%Y%m%d")

# Create Flask app
app = Flask(__name__)


@app.route('/forecast_prices/', methods=['POST'])
def forecast_prices():
    try:
        payload = request.get_json()
        sku = int(payload['sku'])
        time_key = int(payload['time_key'])
        target_date = pd.to_datetime(str(time_key), format="%Y%m%d")
    except Exception:
        return jsonify({"error": "Invalid input format"}), 422

    # Filter data for SKU
    df = prices[prices["sku"] == sku]
    df = df[df["competitor"].isin(["competitorA", "competitorB"])]
    if df.empty:
        return jsonify({"error": "SKU not found"}), 422

    df = df.pivot_table(index="date", columns="competitor", values="final_price").sort_index()
    if "competitorA" not in df.columns or "competitorB" not in df.columns:
        return jsonify({"error": "Missing competitor data"}), 422

    df = df.rename(columns={"competitorA": "price_A", "competitorB": "price_B"})

    # Create lag and rolling features
    history = df[df.index < target_date].copy()
    for lag in [1, 3, 7]:
        history[f"price_A_lag_{lag}"] = history["price_A"].shift(lag)
        history[f"price_B_lag_{lag}"] = history["price_B"].shift(lag)
        history[f"price_A_roll_{lag}"] = history["price_A"].rolling(lag).mean()
        history[f"price_B_roll_{lag}"] = history["price_B"].rolling(lag).mean()

    history = history.dropna()
    if history.empty:
        return jsonify({"error": "Not enough historical data"}), 422

    # Extract last row of features
    features = history.iloc[-1:].drop(columns=["price_A", "price_B"])

    # Predict using both models
    pred_A = float(model_A.predict(features)[0])
    pred_B = float(model_B.predict(features)[0])

    return jsonify({
        "sku": sku,
        "time_key": time_key,
        "pvp_is_competitorA": round(pred_A, 2),
        "pvp_is_competitorB": round(pred_B, 2)
    })


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        return jsonify({'error': f"Observation ID {obs['id']} does not exist"})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([model_to_dict(obs) for obs in Prediction.select()])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
