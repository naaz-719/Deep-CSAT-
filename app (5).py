import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="DeepCSAT Debugged", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_all():
    m = joblib.load('model.joblib')
    s = joblib.load('scaler.joblib')
    f = joblib.load('feature_names.joblib')
    return m, s, f

model, scaler, expected_features = load_all()

st.title("🛡️ DeepCSAT Risk Predictor (Verified)")

# --- 1. DYNAMIC INPUTS ---
# We use the actual feature names to build the UI so there's no mismatch
with st.sidebar:
    st.header("1. Interaction Data")
    # Item Price & Handling Time are direct numericals
    price = st.number_input("Item Price", value=1000.0)
    handling_time = st.number_input("Handling Time (sec)", value=300)
    
    # Dates for Response Time
    t_rep = st.text_input("Reported (DD/MM/YYYY HH:MM)", "01/08/2023 10:00")
    t_res = st.text_input("Responded (DD/MM/YYYY HH:MM)", "01/08/2023 11:30")

    st.header("2. Categories")
    # Matching exact categories from your feature_names.joblib
    chan = st.selectbox("Channel", ["Inbound", "Outcall"])
    cat = st.selectbox("Category", ["Order Related", "Refund Related", "Returns", "Cancellation", "Product Queries"])
    tenure = st.selectbox("Tenure", [">90", "61-90", "31-60", "0-30", "On Job Training"])

st.header("Customer Feedback")
remarks = st.text_area("Customer Remarks", "The agent was very slow and didn't help.")

if st.button("Run Prediction"):
    # --- 2. CALCULATE TRANSFORMED FEATURES ---
    blob = TextBlob(remarks)
    sent_score = blob.sentiment.polarity
    word_count = len(remarks.split())
    
    try:
        fmt = '%d/%m/%Y %H:%M'
        diff = (datetime.strptime(t_res, fmt) - datetime.strptime(t_rep, fmt)).total_seconds() / 60.0
        log_res_time = np.log1p(max(0, diff))
    except:
        log_res_time = 0

    # --- 3. THE FIX: EXACT FEATURE ALIGNMENT ---
    # Create an empty row with all 92 features set to 0
    input_df = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # Fill Numerical Features
    input_df['Item_price'] = price
    input_df['connected_handling_time'] = handling_time
    input_df['Sentiment_Score'] = sent_score
    input_df['Remark_Word_Count'] = word_count
    input_df['Log_Response_Time'] = log_res_time
    
    # Fill Categorical Features (One-Hot Encoding Manual Match)
    # We construct the string exactly how joblib expects it: "Prefix_Value"
    cols_to_set = [
        f"channel_name_{chan}",
        f"category_{cat}",
        f"Tenure Bucket_{tenure}"
    ]
    
    for c in cols_to_set:
        if c in expected_features:
            input_df[c] = 1
        else:
            st.warning(f"Feature '{c}' not found in model schema. Check spelling/casing.")

    # --- 4. PREDICTION ---
    # Scale exactly like training
    scaled_x = scaler.transform(input_df)
    
    prob = model.predict_proba(scaled_x)[0][1]
    pred = 1 if prob > 0.5 else 0 # Explicit threshold

    # --- 5. UI OUTPUT ---
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        if pred == 1:
            st.error(f"### Result: HIGH RISK ({(prob*100):.1f}%)")
            st.write("Customer is likely to give a CSAT 0.")
        else:
            st.success(f"### Result: LOW RISK ({(prob*100):.1f}%)")
            st.write("Customer is likely satisfied.")

    with c2:
        st.write("**Processed Metrics:**")
        st.write(f"- Sentiment: `{sent_score:.2f}`")
        st.write(f"- Log Response Time: `{log_res_time:.2f}`")

    # Debug: Show the active features
    with st.expander("View Model Input Vector (Debug)"):
        st.write(input_df.loc[:, (input_df != 0).any(axis=0)])
