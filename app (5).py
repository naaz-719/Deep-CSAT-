import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="DeepCSAT - Final Predictor", layout="wide")

@st.cache_resource
def load_assets():
    m = joblib.load('model.joblib')
    s = joblib.load('scaler.joblib')
    f = joblib.load('feature_names.joblib')
    return m, s, f

model, scaler, feature_names = load_assets()

# Dynamic Option Extractor
def get_opts(prefix):
    return sorted([c.replace(prefix, "") for c in feature_names if c.startswith(prefix)])

st.title("🛍️ DeepCSAT: Customer Satisfaction Predictor")
st.markdown("Predicting if a customer will give a **Low CSAT (0)** or **High CSAT (1)**.")

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("📊 Transaction Details")
    price = st.number_input("Item Price", value=500.0)
    handling = st.number_input("Handling Time (sec)", value=120)
    
    st.header("🕒 Timing")
    t_rep = st.text_input("Reported (DD/MM/YYYY HH:MM)", "01/08/2023 10:00")
    t_res = st.text_input("Responded (DD/MM/YYYY HH:MM)", "01/08/2023 10:15")

    st.header("📁 Context")
    chan = st.selectbox("Channel", get_opts("channel_name_"))
    cat = st.selectbox("Category", get_opts("category_"))
    sub_cat = st.selectbox("Sub-category", get_opts("Sub-category_"))
    prod = st.selectbox("Product", get_opts("Product_category_"))
    tenure = st.selectbox("Tenure", get_opts("Tenure Bucket_"))

# --- MAIN FEEDBACK ---
st.subheader("💬 Customer Feedback")
remarks = st.text_area("Enter Remarks", "The agent was very helpful, thank you!")

if st.button("Analyze Interaction"):
    # 1. Feature Engineering
    blob = TextBlob(remarks)
    sentiment = blob.sentiment.polarity
    words = len(remarks.split())
    
    try:
        fmt = '%d/%m/%Y %H:%M'
        diff = (datetime.strptime(t_res, fmt) - datetime.strptime(t_rep, fmt)).total_seconds() / 60.0
        log_res = np.log1p(max(0, diff))
    except:
        st.error("Format Error: Use DD/MM/YYYY HH:MM")
        st.stop()

    # 2. Build 92-Column DataFrame
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    input_df['Item_price'] = float(price)
    input_df['connected_handling_time'] = float(handling)
    input_df['Sentiment_Score'] = float(sentiment)
    input_df['Remark_Word_Count'] = float(words)
    input_df['Log_Response_Time'] = float(log_res)
    
    # One-Hot Encoding
    for p, v in [("channel_name_", chan), ("category_", cat), ("Sub-category_", sub_cat), 
                 ("Product_category_", prod), ("Tenure Bucket_", tenure)]:
        col = f"{p}{v}"
        if col in feature_names:
            input_df[col] = 1.0

    # 3. Scaling & Prediction
    input_df = input_df[feature_names] # Strict column ordering
    scaled_x = scaler.transform(input_df)
    
    # Probabilities: [Class 0 (Risk), Class 1 (Satisfied)]
    probs = model.predict_proba(scaled_x)[0]
    prob_risk = probs[0]
    prob_satisfied = probs[1]

    # --- BALANCED LOGIC ---
    # We use a 0.6 threshold instead of 0.5 to prevent "false alarms" on minor delays
    is_risk = (prob_risk > 0.6)
    
    # SENTIMENT VETO: If the text is clearly positive (>0.3), override minor risks
    if sentiment > 0.3 and prob_risk < 0.85:
        is_risk = False

    # --- RESULTS ---
    st.divider()
    if is_risk:
        st.error(f"### 🚩 RESULT: HIGH RISK (Low CSAT Likely)")
        st.write(f"Model Risk Confidence: {prob_risk*100:.1f}%")
        st.write("**Warning:** Features like response delay or channel may be driving this risk.")
    else:
        st.success(f"### ✅ RESULT: LOW RISK (Satisfied)")
        st.write(f"Model Satisfaction Confidence: {prob_satisfied*100:.1f}%")

    # --- DEBUG DATA ---
    with st.expander("🔬 Technical Breakdown"):
        st.write(f"**Detected Sentiment:** {sentiment:.2f}")
        st.write(f"**Log Response Time:** {log_res:.2f}")
        st.write("**Active Features in Model:**")
        st.table(input_df.loc[:, (input_df != 0).any(axis=0)].T)

