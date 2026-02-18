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

# Extract clean options from the 92 features
def get_options(prefix):
    return sorted([c.replace(prefix, "") for c in feature_names if c.startswith(prefix)])

st.set_page_config(page_title="DeepCSAT Debugger", layout="wide")
st.title("🛡️ DeepCSAT Prediction Engine")

# --- UI INPUTS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Numeric Inputs")
    price = st.number_input("Item Price", value=25000.0)
    handling = st.number_input("Handling Time (Seconds)", value=1200) # High value helps trigger risk
    
    st.subheader("Response Time Logic")
    # Using text input to ensure users provide the full format
    t_rep = st.text_input("Reported (DD/MM/YYYY HH:MM)", "01/08/2023 10:00")
    t_res = st.text_input("Responded (DD/MM/YYYY HH:MM)", "05/08/2023 10:00")

with col2:
    st.subheader("Categories")
    chan = st.selectbox("Channel", get_options("channel_name_"))
    cat = st.selectbox("Category", get_options("category_"))
    sub_cat = st.selectbox("Sub-category", get_options("Sub-category_"))
    prod = st.selectbox("Product", get_options("Product_category_"))
    tenure = st.selectbox("Tenure Bucket", get_options("Tenure Bucket_"))

remarks = st.text_area("Customer Remarks", "EXTREMELY DISAPPOINTED. No one helped me for days!")

if st.button("Run Prediction"):
    # 1. Feature Engineering
    blob = TextBlob(remarks)
    sentiment = blob.sentiment.polarity
    word_count = len(remarks.split())
    
    try:
        fmt = '%d/%m/%Y %H:%M'
        dt_diff = (datetime.strptime(t_res, fmt) - datetime.strptime(t_rep, fmt)).total_seconds() / 60.0
        log_res = np.log1p(max(0, dt_diff))
    except:
        st.error("Date Format Error! Use DD/MM/YYYY HH:MM")
        st.stop()

    # 2. Alignment (The 92-Feature Vector)
    input_df = pd.DataFrame(0.0, index=[0], columns=feature_names)
    input_df['Item_price'] = float(price)
    input_df['connected_handling_time'] = float(handling)
    input_df['Sentiment_Score'] = float(sentiment)
    input_df['Remark_Word_Count'] = float(word_count)
    input_df['Log_Response_Time'] = float(log_res)
    
    # One-Hot Encoding
    for p, v in [("channel_name_", chan), ("category_", cat), ("Sub-category_", sub_cat), 
                 ("Product_category_", prod), ("Tenure Bucket_", tenure)]:
        col = f"{p}{v}"
        if col in feature_names:
            input_df[col] = 1.0

    # 3. Predict
    input_df = input_df[feature_names] # Ensure strict order
    scaled_x = scaler.transform(input_df)
    prob = model.predict_proba(scaled_x)[0] # [Prob_0, Prob_1]
    
    # --- OUTPUT ---
    st.divider()
    # Class 0 = LOW CSAT (High Risk), Class 1 = HIGH CSAT (Safe)
    if prob[0] > 0.5:
        st.error(f"### 🚩 HIGH RISK (Score 0) | Confidence: {prob[0]*100:.1f}%")
    else:
        st.success(f"### ✅ LOW RISK (Score 1) | Confidence: {prob[1]*100:.1f}%")

    with st.expander("🔬 Why this result? (Feature Breakdown)"):
        st.write(f"**Sentiment:** {sentiment:.2f} (Negative is bad)")
        st.write(f"**Log Response Time:** {log_res:.2f} (Higher is worse)")
        st.write("**Columns Sent to Model:**")
        st.dataframe(input_df.loc[:, (input_df != 0).any(axis=0)])
