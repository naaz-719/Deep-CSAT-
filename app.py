import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob

# Set page configuration
st.set_page_config(page_title="DeepCSAT - Risk Predictor", layout="wide")

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, scaler, and feature names."""
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    features = joblib.load('feature_names.joblib')
    return model, scaler, features

try:
    model, scaler, feature_names = load_model_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def get_sentiment(text):
    """Calculate sentiment score from customer remarks."""
    if not text or str(text).lower() == 'nan':
        return 0.0
    return TextBlob(str(text)).sentiment.polarity

def preprocess_input(input_data, feature_names, scaler):
    """Match input data to training features and scale."""
    # Create an empty dataframe with all training features
    df_input = pd.DataFrame(columns=feature_names)
    df_input.loc[0] = 0  # Initialize with zeros
    
    # Fill in numerical/calculated values
    df_input['Item_price'] = input_data['Item_price']
    df_input['Sentiment_Score'] = get_sentiment(input_data['Customer Remarks'])
    df_input['Remark_Word_Count'] = len(str(input_data['Customer Remarks']).split())
    
    # One-Hot Encoding (Manual mapping to match feature_names)
    categorical_mappings = {
        f"channel_name_{input_data['channel_name']}": 1,
        f"category_{input_data['category']}": 1,
        f"Sub-category_{input_data['Sub-category']}": 1,
        f"Tenure Bucket_{input_data['Tenure Bucket']}": 1
    }
    
    for feature, value in categorical_mappings.items():
        if feature in df_input.columns:
            df_input[feature] = value

    # Scale the features
    scaled_data = scaler.transform(df_input)
    return scaled_data

# --- UI LAYOUT ---
st.title("📊 DeepCSAT: Customer Satisfaction Risk Predictor")
st.markdown("Predict if a customer interaction is **High Risk** (potential CSAT score of 0) in real-time.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        channel = st.selectbox("Channel Name", ["Inbound", "Outcall", "Email"])
        category = st.selectbox("Category", ["Order Related", "Returns", "Cancellation", "Refund Related", "Product Queries", "Others"])
        sub_category = st.text_input("Sub-category (e.g., Installation/demo)", "General Enquiry")
        tenure = st.selectbox("Tenure Bucket", ["0-30", "31-60", "61-90", ">90", "On Job Training"])
    
    with col2:
        price = st.number_input("Item Price", min_value=0.0, value=100.0)
        remarks = st.text_area("Customer Remarks", placeholder="Enter customer feedback here...")
    
    submit = st.form_submit_button("Predict CSAT Risk")

# --- PREDICTION LOGIC ---
if submit:
    input_payload = {
        'channel_name': channel,
        'category': category,
        'Sub-category': sub_category,
        'Tenure Bucket': tenure,
        'Item_price': price,
        'Customer Remarks': remarks
    }
    
    # Preprocess and predict
    processed_data = preprocess_input(input_payload, feature_names, scaler)
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    # Display results
    st.divider()
    if prediction == 0:
        st.error(f"### ⚠️ HIGH RISK DETECTED")
        st.write(f"Confidence of low CSAT: {probability:.2%}")
        st.write("Recommendation: Escalate to a senior manager immediately.")
    else:
        st.success(f"### ✅ LOW RISK")
        st.write(f"Probability of satisfaction: {1 - probability:.2%}")

