import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="DeepCSAT Predictor", page_icon="🛍️", layout="wide")

# --- LOAD ASSETS ---
@st.cache_resource
def load_model_artifacts():
    # Loading the specific files you uploaded
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    features = joblib.load('feature_names.joblib')
    return model, scaler, features

try:
    model, scaler, expected_features = load_model_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- UI HEADER ---
st.title("🛍️ DeepCSAT: Ecommerce Satisfaction Predictor")
st.markdown("""
This app predicts the risk of a customer providing a **Low CSAT Score (0)** based on real-time interaction data.
""")

# --- INPUT FORM ---
with st.sidebar:
    st.header("Input Interaction Details")
    
    channel = st.selectbox("Channel", ["Inbound", "Outcall"])
    category = st.selectbox("Category", ["Order Related", "Refund Related", "Returns", "Cancellation", "Product Queries", "Others"])
    sub_category = st.text_input("Sub-category", "Order status enquiry")
    product_cat = st.selectbox("Product Category", ["Mobile", "Electronics", "LifeStyle", "Home", "Books", "UNKNOWN"])
    tenure = st.selectbox("Tenure Bucket", [">90", "61-90", "31-60", "0-30", "On Job Training"])
    
    price = st.number_input("Item Price", min_value=0.0, value=25000.0)
    handling_time = st.number_input("Handling Time (sec)", min_value=0, value=300)
    
    # Timestamps for response time calculation
    reported_time = st.text_input("Issue Reported (DD/MM/YYYY HH:MM)", datetime.now().strftime("%d/%m/%Y %H:%M"))
    responded_time = st.text_input("Issue Responded (DD/MM/YYYY HH:MM)", datetime.now().strftime("%d/%m/%Y %H:%M"))

st.subheader("Customer Feedback Analysis")
remarks = st.text_area("Customer Remarks", placeholder="Enter customer comments here...")

# --- PREDICTION LOGIC ---
if st.button("Predict CSAT Risk"):
    # 1. Feature Engineering: Sentiment
    blob = TextBlob(remarks)
    sentiment_score = blob.sentiment.polarity
    word_count = len(remarks.split())
    
    # 2. Feature Engineering: Log Response Time
    try:
        fmt = '%d/%m/%Y %H:%M'
        t1 = datetime.strptime(reported_time, fmt)
        t2 = datetime.strptime(responded_time, fmt)
        diff_mins = (t2 - t1).total_seconds() / 60.0
        log_response_time = np.log1p(max(0, diff_mins))
    except:
        log_response_time = 0
        st.warning("Date format error. Using default response time.")

    # 3. Create DataFrame for encoding
    input_dict = {
        'Item_price': price,
        'connected_handling_time': handling_time,
        'Sentiment_Score': sentiment_score,
        'Remark_Word_Count': word_count,
        'Log_Response_Time': log_response_time,
        f'channel_name_{channel}': 1,
        f'category_{category}': 1,
        f'Sub-category_{sub_category}': 1,
        f'Product_category_{product_cat}': 1,
        f'Tenure Bucket_{tenure}': 1
    }
    
    # 4. Align with the 92 expected features
    # This ensures the input matches the model's exact training structure
    final_features = pd.DataFrame(columns=expected_features)
    row = pd.Series(0, index=expected_features)
    
    for key, value in input_dict.items():
        if key in expected_features:
            row[key] = value
            
    final_features.loc[0] = row
    
    # 5. Scale and Predict
    scaled_data = scaler.transform(final_features)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]

    # --- RESULTS DISPLAY ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sentiment Score", f"{sentiment_score:.2f}")
    with col2:
        st.metric("Risk Probability", f"{probability*100:.1f}%")
    with col3:
        status = "🔴 HIGH RISK (Likely CSAT 0)" if prediction == 1 else "🟢 LOW RISK (Satisfied)"
        st.subheader(status)

    if prediction == 1:
        st.error("Action Required: This interaction shows high markers of dissatisfaction. Consider immediate supervisor intervention.")
    else:
        st.success("Interaction appears healthy. Model predicts a positive CSAT outcome.")

    # Show a small chart of feature contributions (Optional)
    st.write("### Model Insight")
    st.bar_chart(pd.DataFrame({
        'Metric': ['Sentiment', 'Log Response Time', 'Handling Time'],
        'Value': [sentiment_score, log_response_time, (handling_time/1000)]
    }).set_index('Metric'))
