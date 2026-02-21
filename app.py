import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob

# --- PAGE CONFIG ---
st.set_page_config(page_title="DeepCSAT Risk Predictor", layout="centered")

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_assets():
    # Using the 'best' model as requested
    model = joblib.load('best_csat_predictor_model.joblib')
    scaler = joblib.load('scaler.joblib')
    features = joblib.load('feature_names.joblib')
    return model, scaler, features

try:
    model, scaler, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---
def get_sentiment(text):
    """Calculates Sentiment Score using TextBlob."""
    if not text or str(text).strip() == "":
        return 0.0
    return TextBlob(str(text)).sentiment.polarity

def preprocess_payload(data, features, scaler):
    """Aligns input data with the 92 training features and scales them."""
    # Create base dataframe with zeros for all 92 features
    df_input = pd.DataFrame(columns=features)
    df_input.loc[0] = 0
    
    # 1. Fill Numerical & Calculated Fields
    df_input['Item_price'] = data['Item_price']
    df_input['Sentiment_Score'] = get_sentiment(data['Customer Remarks'])
    df_input['Remark_Word_Count'] = len(str(data['Customer Remarks']).split())
    # Note: Placeholder values for training-only metrics if not available in real-time
    df_input['connected_handling_time'] = data.get('handling_time', 0)
    df_input['Log_Response_Time'] = np.log1p(data.get('response_time', 0))

    # 2. Map Categorical Fields (One-Hot Encoding)
    mappings = [
        f"channel_name_{data['channel']}",
        f"category_{data['category']}",
        f"Sub-category_{data['sub_category']}",
        f"Tenure Bucket_{data['tenure']}"
    ]
    
    for col in mappings:
        if col in df_input.columns:
            df_input[col] = 1

    # 3. Scale and Return
    return scaler.transform(df_input)

# --- USER INTERFACE ---
st.title("🛡️ DeepCSAT Early Warning System")
st.markdown("Predict customer dissatisfaction risk before the survey is even sent.")

with st.form("risk_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        channel = st.selectbox("Channel", ["Inbound", "Outcall", "Email"])
        category = st.selectbox("Category", ["Order Related", "Returns", "Cancellation", "Refund Related", "Product Queries"])
        sub_cat = st.text_input("Sub-Category", "General Enquiry")
        tenure = st.selectbox("Tenure Bucket", ["0-30", "31-60", "61-90", ">90", "On Job Training"])

    with col2:
        price = st.number_input("Item Price", min_value=0.0, value=50.0)
        resp_time = st.number_input("Response Time (Minutes)", min_value=0, value=15)
        remarks = st.text_area("Customer Remarks", placeholder="Type customer feedback...")

    submit = st.form_submit_button("Analyze CSAT Risk")

# --- PREDICTION LOGIC ---
if submit:
    input_data = {
        'channel': channel, 'category': category, 'sub_category': sub_cat,
        'tenure': tenure, 'Item_price': price, 'Customer Remarks': remarks,
        'response_time': resp_time
    }
    
    processed_x = preprocess_payload(input_data, feature_names, scaler)
    prediction = model.predict(processed_x)[0]
    prob = model.predict_proba(processed_x)[0][1]

    st.divider()
    if prediction == 0:  # Assuming 0 is the 'High Risk' class based on project summary
        st.error(f"### 🚩 HIGH RISK DETECTED")
        st.write(f"Confidence: **{prob:.1%}**")
        st.warning("Recommendation: Proactively escalate to a Senior Manager.")
    else:
        st.success(f"### ✅ LOW RISK")
        st.write(f"Customer is likely to be satisfied (Prob: {1-prob:.1%})")
