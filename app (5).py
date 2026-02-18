import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
from datetime import datetime

# --- LOAD ASSETS ---
@st.cache_resource
def load_all():
    m = joblib.load('model.joblib')
    s = joblib.load('scaler.joblib')
    f = joblib.load('feature_names.joblib')
    return m, s, f

model, scaler, feature_names = load_all()

def get_options(prefix):
    return sorted([c.replace(prefix, "") for c in feature_names if c.startswith(prefix)])

st.title("🛡️ DeepCSAT Prediction Engine (V2 - Logic Fix)")

# --- INPUTS ---
col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Item Price", value=500.0)
    handling = st.number_input("Handling Time (Sec)", value=300)
    t_rep = st.text_input("Reported At", "01/08/2023 10:00")
    t_res = st.text_input("Responded At", "01/08/2023 10:30")

with col2:
    chan = st.selectbox("Channel", get_options("channel_name_"))
    cat = st.selectbox("Category", get_options("category_"))
    sub_cat = st.selectbox("Sub-category", get_options("Sub-category_"))
    prod = st.selectbox("Product", get_options("Product_category_"))
    tenure = st.selectbox("Tenure", get_options("Tenure Bucket_"))

remarks = st.text_area("Customer Remarks", "Excellent service, very happy!")

if st.button("Run Prediction"):
    # 1. Feature Engineering
    blob = TextBlob(remarks)
    sentiment = blob.sentiment.polarity
    word_count = len(remarks.split())
    
    try:
        fmt = '%d/%m/%Y %H:%M'
        diff = (datetime.strptime(t_res, fmt) - datetime.strptime(t_rep, fmt)).total_seconds() / 60.0
        log_res = np.log1p(max(0, diff))
    except:
        log_res = 0

    # 2. Vectorization
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    input_df['Item_price'] = float(price)
    input_df['connected_handling_time'] = float(handling)
    input_df['Sentiment_Score'] = float(sentiment)
    input_df['Remark_Word_Count'] = float(word_count)
    input_df['Log_Response_Time'] = float(log_res)
    
    for p, v in [("channel_name_", chan), ("category_", cat), ("Sub-category_", sub_cat), 
                 ("Product_category_", prod), ("Tenure Bucket_", tenure)]:
        col = f"{p}{v}"
        if col in feature_names: input_df[col] = 1.0

    # 3. Scaling & Prediction
    input_df = input_df[feature_names] 
    scaled_x = scaler.transform(input_df)
    
    # Get raw probabilities
    probs = model.predict_proba(scaled_x)[0]
    # In your model: Index 0 = Risk (CSAT 0), Index 1 = Satisfied (CSAT 1)
    prob_risk = probs[0]
    prob_satisfied = probs[1]

    # --- RESULT LOGIC ---
    st.divider()
    
    # If sentiment is very positive (>0.5), force it to check why it's failing
    if sentiment > 0.5 and prob_risk > 0.5:
        st.warning("⚠️ Sentiment is POSITIVE but Model predicts RISK. Check features below.")

    if prob_risk > prob_satisfied:
        st.error(f"### 🚩 RESULT: HIGH RISK (Likely CSAT 0)")
        st.write(f"Confidence in Risk: {prob_risk*100:.1f}%")
    else:
        st.success(f"### ✅ RESULT: SATISFIED (Likely CSAT 1)")
        st.write(f"Confidence in Satisfaction: {prob_satisfied*100:.1f}%")

    # --- DEBUGGING THE "WRONG" OUTPUT ---
    with st.expander("🔬 View Model Decision Factors"):
        st.write(f"**Detected Sentiment:** {sentiment:.2f} (1.0 is Best, -1.0 is Worst)")
        st.write(f"**Log Response Delay:** {log_res:.2f}")
        st.write("**Features influencing the model:**")
        active = input_df.loc[:, (input_df != 0).any(axis=0)]
        st.table(active.T)

