"""Shared inference helpers used by the inference notebook and the Streamlit app."""
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "purchase_intent_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"


def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def feature_list(meta):
    f = meta["features"]
    return f["numeric"] + f["binary"] + f["categorical"]


def segment(p, threshold):
    """Translate a probability into an action the marketing team can use."""
    if p >= threshold:
        return "High intent", "Likely to buy — don't give away a discount; show free-shipping threshold / installment banner."
    if p >= threshold / 2:
        return "Medium intent", "Hesitating — best target for a coupon, cart reminder via WhatsApp/e-mail."
    return "Low intent", "Browsing — show content, reviews and social proof; no discount."


def predict_df(model, meta, df):
    """Score a DataFrame of sessions. Returns a copy with probability, prediction and segment."""
    missing = [c for c in feature_list(meta) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    out = df.copy()
    thr = meta["threshold"]
    out["purchase_probability"] = model.predict_proba(df[feature_list(meta)])[:, 1].round(4)
    out["predicted_purchase"] = (out["purchase_probability"] >= thr).astype(int)
    seg = out["purchase_probability"].map(lambda p: segment(p, thr))
    out["intent_segment"] = seg.str[0]
    out["recommended_action"] = seg.str[1]
    return out
