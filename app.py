"""
Purchase Intent Predictor (Streamlit)
Brazilian e-commerce client (name withheld) · synthetic data

Run locally:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import feature_list, load_model, predict_df

ROOT = Path(__file__).parent
IMG = ROOT / "images"

st.set_page_config(page_title="Purchase Intent · Brazilian E-commerce", page_icon="🛒", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_data():
    return pd.read_csv(ROOT / "data" / "processed" / "sessions_clean.csv", parse_dates=["session_start"])


model, meta = get_model()
df = get_data()
cats = meta["categories"]

page = st.sidebar.radio("Navigation", ["Overview", "Explore the data", "Predict a session", "Score a CSV file"])
st.sidebar.markdown("---")
st.sidebar.caption("Client name withheld · synthetic data.")

# ------------------------------------------------------------------ overview
if page == "Overview":
    st.title("🛒 Will this visitor buy?")
    st.markdown(
        "**Client:** a mid-sized Brazilian online retailer of fashion and home goods (name withheld), "
        "based in São Paulo.  \n**Problem:** traffic keeps growing but only ~1 in 6 sessions converts. "
        "Marketing wants to know **which live sessions are likely to buy**, so they can spend coupons and "
        "WhatsApp follow-ups only where they make a difference."
    )
    m = meta["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sessions analysed", f"{len(df):,}")
    c2.metric("Conversion rate", f"{df['purchased'].mean():.1%}")
    c3.metric("Model F1 (test)", f"{m['f1']:.2f}")
    c4.metric("Buyers reached in top 20%", f"{meta['cumulative_gains']['top_20pct']:.0%}")
    st.image(str(IMG / "13_cumulative_gains.png"), width=560)
    st.markdown(
        "- **Model:** tuned Histogram Gradient Boosting inside a scikit-learn pipeline\n"
        f"- **Test set:** precision {m['precision']:.2f} · recall {m['recall']:.2f} · PR-AUC {m['pr_auc']:.2f} · ROC-AUC {m['roc_auc']:.2f}\n"
        f"- **Decision threshold:** {meta['threshold']} (chosen to maximise F1 on cross-validated training predictions)"
    )

# ------------------------------------------------------------------ EDA
elif page == "Explore the data":
    st.title("📊 Explore the sessions")
    dim_labels = {"Traffic source": "traffic_source", "Device": "device_type", "Region": "region",
                  "Sales event": "sales_event", "Visitor type": "visitor_type", "Hour of day": "hour"}
    c1, c2 = st.columns([1, 3])
    with c1:
        dim = st.selectbox("Break conversion down by", list(dim_labels))
        devices = st.multiselect("Device filter", sorted(df["device_type"].unique()),
                                 default=sorted(df["device_type"].unique()))
    sub = df[df["device_type"].isin(devices)]
    g = (sub.groupby(dim_labels[dim])["purchased"].agg(conversion_rate="mean", sessions="size")
         .sort_values("conversion_rate", ascending=False))
    with c2:
        st.bar_chart(g["conversion_rate"], horizontal=True, color="#2a78d6")
    st.dataframe(g.style.format({"conversion_rate": "{:.1%}", "sessions": "{:,}"}), width="stretch")

    st.subheader("Key findings")
    tabs = st.tabs(["Micro-conversions", "Calendar", "Trend", "Feature importance"])
    for tab, img, text in zip(tabs,
                              ["07_micro_conversions.png", "05_conv_by_event.png", "06_monthly_trend.png",
                               "12_feature_importance.png"],
                              ["Checking **frete** or **parcelamento** triples the chance of buying.",
                               "Black Friday and Natal lift conversion ~40% vs. a regular day; Carnaval is the low point.",
                               "December is the peak month for both traffic and conversion.",
                               "Page value and cart activity dominate, followed by the freight and installment signals."]):
        with tab:
            st.markdown(text)
            st.image(str(IMG / img), width=720)

# ------------------------------------------------------------------ single prediction
elif page == "Predict a session":
    st.title("🔮 Predict a single session")
    with st.form("session"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Visitor**")
            traffic_source = st.selectbox("Traffic source", [c for c in cats["traffic_source"] if c != "Unknown"])
            device_type = st.selectbox("Device", [c for c in cats["device_type"] if c != "Unknown"])
            visitor_type = st.selectbox("Visitor type", [c for c in cats["visitor_type"] if c != "Unknown"])
            region = st.selectbox("Region", [c for c in cats["region"] if c != "Unknown"])
            sales_event = st.selectbox("Sales event", cats["sales_event"], index=cats["sales_event"].index("Regular day"))
            month = st.slider("Month", 1, 12, 11)
            hour = st.slider("Hour of day", 0, 23, 20)
            is_weekend = st.checkbox("Weekend")
        with c2:
            st.markdown("**Browsing**")
            product_pages = st.number_input("Product pages", 0, 100, 5)
            category_pages = st.number_input("Category pages", 0, 50, 2)
            searches = st.number_input("Searches", 0, 30, 1)
            session_duration_sec = st.number_input("Session duration (sec)", 0, 10_000, 300)
            product_page_duration_sec = st.number_input("Time on product pages (sec)", 0, 10_000, 120)
            bounce_rate = st.slider("Bounce rate", 0.0, 1.0, 0.15)
            exit_rate = st.slider("Exit rate", 0.0, 1.0, 0.28)
        with c3:
            st.markdown("**Shopping actions**")
            add_to_cart_count = st.number_input("Items added to cart", 0, 20, 0)
            cart_value_brl = st.number_input("Cart value (R$)", 0.0, 20_000.0, 0.0, step=10.0)
            wishlist_adds = st.number_input("Wishlist adds", 0, 20, 0)
            page_value_brl = st.number_input("Page value (R$)", 0.0, 1_000.0, 0.0, step=5.0)
            viewed_freight_calculator = st.checkbox("Used freight calculator (frete)")
            viewed_installment_options = st.checkbox("Viewed installment options (parcelamento)")
            coupon_applied = st.checkbox("Applied a coupon")
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "pages_viewed": 1 + product_pages + category_pages, "category_pages": category_pages,
            "product_pages": product_pages, "searches": searches, "session_duration_sec": session_duration_sec,
            "product_page_duration_sec": min(product_page_duration_sec, session_duration_sec),
            "add_to_cart_count": add_to_cart_count, "wishlist_adds": wishlist_adds, "cart_value_brl": cart_value_brl,
            "bounce_rate": bounce_rate, "exit_rate": exit_rate, "page_value_brl": page_value_brl,
            "hour": hour, "month": month, "viewed_freight_calculator": int(viewed_freight_calculator),
            "viewed_installment_options": int(viewed_installment_options), "coupon_applied": int(coupon_applied),
            "is_weekend": int(is_weekend), "device_type": device_type, "traffic_source": traffic_source,
            "visitor_type": visitor_type, "region": region, "sales_event": sales_event,
        }])
        res = predict_df(model, meta, row).iloc[0]
        p = res["purchase_probability"]
        c1, c2 = st.columns([1, 2])
        c1.metric("Purchase probability", f"{p:.0%}")
        c1.progress(float(p))
        (c2.success if res["intent_segment"] == "High intent" else
         c2.warning if res["intent_segment"] == "Medium intent" else c2.info)(
            f"**{res['intent_segment']}** — {res['recommended_action']}")

# ------------------------------------------------------------------ batch scoring
else:
    st.title("📁 Score a CSV export")
    st.markdown(f"Upload a CSV with these columns: `{', '.join(feature_list(meta))}`.")
    sample = (ROOT / "data" / "sample" / "new_sessions.csv").read_bytes()
    st.download_button("Download a sample file", sample, "new_sessions.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            scored = predict_df(model, meta, pd.read_csv(up))
        except ValueError as e:
            st.error(str(e))
        else:
            st.write(scored["intent_segment"].value_counts())
            st.dataframe(scored.sort_values("purchase_probability", ascending=False), width="stretch")
            st.download_button("Download scored file", scored.to_csv(index=False).encode(), "scored_sessions.csv",
                               "text/csv")
