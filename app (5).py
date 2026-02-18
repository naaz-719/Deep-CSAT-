import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
from datetime import datetime

# --- LOAD ASSETS ---
@st.cache_resource
def load_all():
    # Load your specific uploaded files
    m = joblib.load('model.joblib')
    s = joblib.load('scaler.joblib')
    f = joblib.load('feature_names.joblib')
    return m, s, f

try:
    model, scaler, feature_names = load_all()
except Exception as e:
    st.error(f"Failed to load joblib files: {e}")
    st.stop()

# --- DYNAMIC CATEGORY EXTRACTOR ---
def get_cats(prefix):
    return sorted([c.replace(prefix, "") for c in feature_names if c.startswith(prefix)])

st.title("🛡️ DeepCSAT Production Predictor")

# --- UI LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Numeric Metrics")
    price = st.number_input("Item Price", value=1500.0)
    handling = st.number_input("Handling Time (Seconds)", value=300)
    
    st.subheader("🕒 Timing")
    t_rep = st.text_input("Issue Reported (DD/MM/YYYY HH:MM)", "01/08/2023 10:00")
    t_res = st.text_input("Issue Responded (DD/MM/YYYY HH:MM)", "01/08/2023 12:00")

with col2:
    st.subheader("📂 Categorization")
    chan = st.selectbox("Channel", get_cats("channel_name_"))
    cat = st.selectbox("Category", get_cats("category_"))
    sub_cat = st.selectbox("Sub-category", get_cats("Sub-category_"))
    prod = st.selectbox("Product", get_cats("Product_category_"))
    tenure = st.selectbox("Tenure", get_cats("Tenure Bucket_"))

st.subheader("💬 Feedback")
remarks = st.text_area("Customer Remarks", "The agent was unhelpful and I am still waiting.")

if st.button("🚀 Predict CSAT Risk"):
    try:
        # 1. Feature Engineering (As per Notebook)
        blob = TextBlob(remarks)
        sent = blob.sentiment.polarity
        words = len(remarks.split())
        
        # Date Difference
        fmt = '%d/%m/%Y %H:%M'
        dt_diff = (datetime.strptime(t_res, fmt) - datetime.strptime(t_rep, fmt)).total_seconds() / 60.0
        log_res = np.log1p(max(0, dt_diff))

        # 2. Build 92-Column Input
        # We initialize with Zeros to ensure all OHE columns exist
        input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
        
        # Map Numerics
        input_df['Item_price'] = float(price)
        input_df['connected_handling_time'] = float(handling)
        input_df['Sentiment_Score'] = float(sent)
        input_df['Remark_Word_Count'] = float(words)
        input_df['Log_Response_Time'] = float(log_res)
        
        # Map One-Hot Categories
        for prefix, val in [("channel_name_", chan), ("category_", cat), 
                            ("Sub-category_", sub_cat), ("Product_category_", prod), 
                            ("Tenure Bucket_", tenure)]:
            col_name = f"{prefix}{val}"
            if col_name in feature_names:
                input_df[col_name] = 1.0

        # 3. CRITICAL STEP: Reorder columns to match exactly what scaler expects
        input_df = input_df[feature_names]

        # 4. Scale and Predict
        scaled_x = scaler.transform(input_df)
        prob = model.predict_proba(scaled_x)[0] # [Prob_0, Prob_1]
        pred = model.predict(scaled_x)[0]

        # 5. Output Results
        st.divider()
        # In the notebook: Class 0 = Low CSAT (Risk), Class 1 = High CSAT (Safe)
        if pred == 0:
            st.error(f"### 🚩 HIGH RISK (Score 0) | Confidence: {prob[0]*100:.1f}%")
            st.warning("Customer is likely to be dissatisfied.")
        else:
            st.success(f"### ✅ LOW RISK (Score 1) | Confidence: {prob[1]*100:.1f}%")
            st.info("Customer interaction looks healthy.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.info("Check if your date format is exactly DD/MM/YYYY HH:MM (e.g., 01/08/2023 14:00)")
