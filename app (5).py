import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
from datetime import datetime

# --- SETTINGS ---
st.set_page_config(page_title="DeepCSAT – Risk Predictor", layout="wide")

@st.cache_resource
def load_assets():
    m = joblib.load('model.joblib')
    s = joblib.load('scaler.joblib')
    f = joblib.load('feature_names.joblib')
    return m, s, f

model, scaler, feature_names = load_assets()

# Helper to get clean categories from the joblib list
def get_options(prefix):
    return sorted([c.replace(prefix, "") for c in feature_names if c.startswith(prefix)])

# --- UI ---
st.title("🛡️ DeepCSAT Prediction Engine")
st.markdown("Identifies high-risk interactions (Target 0) vs Low-risk (Target 1).")

with st.sidebar:
    st.header("📋 Interaction Metrics")
    price = st.number_input("Item Price", value=500.0)
    handling = st.number_input("Handling Time (Seconds)", value=120)
    
    st.header("🕒 Response Time")
    t_rep = st.date_input("Reported Date", datetime(2023, 8, 1))
    t_rep_time = st.time_input("Reported Time", datetime.strptime("10:00", "%H:%M").time())
    t_res = st.date_input("Responded Date", datetime(2023, 8, 2))
    t_res_time = st.time_input("Responded Time", datetime.strptime("10:00", "%H:%M").time())

    st.header("📁 Categorization")
    chan = st.selectbox("Channel", get_options("channel_name_"))
    cat = st.selectbox("Category", get_options("category_"))
    sub_cat = st.selectbox("Sub-Category", get_options("Sub-category_"))
    prod = st.selectbox("Product", get_options("Product_category_"))
    tenure = st.selectbox("Tenure", get_options("Tenure Bucket_"))

st.subheader("💬 Customer Feedback")
remarks = st.text_area("Customer Remarks", placeholder="Enter the exact customer comment...")

if st.button("Analyze Risk"):
    # 1. Calculation Logic (Matching Notebook)
    blob = TextBlob(remarks)
    sentiment = blob.sentiment.polarity
    word_count = len(remarks.split())
    
    # Calculate Response Time (Minutes)
    dt_rep = datetime.combine(t_rep, t_rep_time)
    dt_res = datetime.combine(t_res, t_res_time)
    diff_minutes = (dt_res - dt_rep).total_seconds() / 60
    log_response_time = np.log1p(max(0, diff_minutes))

    # 2. Build the exact 92-feature vector
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Numerical mapping
    input_df['Item_price'] = price
    input_df['connected_handling_time'] = handling
    input_df['Sentiment_Score'] = sentiment
    input_df['Remark_Word_Count'] = word_count
    input_df['Log_Response_Time'] = log_response_time
    
    # Categorical mapping (One-Hot)
    for col, val in [("channel_name_", chan), ("category_", cat), 
                     ("Sub-category_", sub_cat), ("Product_category_", prod), 
                     ("Tenure Bucket_", tenure)]:
        full_col = f"{col}{val}"
        if full_col in feature_names:
            input_df[full_col] = 1

    # 3. Scaling & Prediction
    # CRITICAL: We must re-order columns to match the scaler's original training order
    input_df = input_df[feature_names] 
    
    scaled_data = scaler.transform(input_df)
    
    # Risk calculation
    prob = model.predict_proba(scaled_data)[0] # [Prob_0, Prob_1]
    prediction = model.predict(scaled_data)[0]

    # --- DISPLAY ---
    st.divider()
    
    # In your notebook: 0 = Low CSAT (Risk), 1 = High CSAT (Safe)
    if prediction == 0:
        st.error(f"### 🚩 HIGH RISK DETECTED ({(prob[0]*100):.1f}%)")
        st.write("The model predicts this customer will provide a **Low CSAT score (0)**.")
    else:
        st.success(f"### ✅ LOW RISK ({(prob[1]*100):.1f}%)")
        st.write("The model predicts a **Satisfied outcome**.")

    # --- DEBUG PANEL ---
    with st.expander("🔍 Deep Debug: Why this output?"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Processed Features:**")
            st.write(f"- Sentiment: `{sentiment:.2f}`")
            st.write(f"- Log Resp Time: `{log_response_time:.2f}`")
            st.write(f"- Word Count: `{word_count}`")
        with col_b:
            st.write("**Top Active Features:**")
            active = input_df.loc[:, (input_df != 0).any(axis=0)]
            st.table(active.T)
