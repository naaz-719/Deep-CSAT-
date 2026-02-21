import streamlit as st
import pandas as pd
import numpy as np
import joblib
from textblob import TextBlob
import nltk

# --- NLTK SETUP ---
# Required for TextBlob to function correctly in cloud environments
@st.cache_resource
def download_nltk_resources():
    try:
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

download_nltk_resources()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="DeepCSAT - Risk Predictor",
    page_icon="🛡️",
    layout="wide"
)

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_ml_components():
    """Loads the model, scaler, and feature names."""
    try:
        model = joblib.load('best_csat_predictor_model.joblib')
        scaler = joblib.load('scaler.joblib')
        features = joblib.load('feature_names.joblib')
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"Missing file: {e}")
        st.stop()

model, scaler, feature_names = load_ml_components()

# --- HELPER FUNCTIONS ---
def get_sentiment(text):
    """Calculates sentiment polarity (-1 to 1)."""
    if not text or str(text).strip() == "":
        return 0.0
    return TextBlob(str(text)).sentiment.polarity

def preprocess_input(data, feature_names, scaler):
    """Aligns user input with the 92 training features."""
    # Create empty DataFrame with all 92 columns set to 0
    df_input = pd.DataFrame(columns=feature_names)
    df_input.loc[0] = 0
    
    # 1. Fill Numerical & Engineered Features
    df_input['Item_price'] = data['Item_price']
    df_input['Sentiment_Score'] = get_sentiment(data['remarks'])
    df_input['Remark_Word_Count'] = len(str(data['remarks']).split())
    df_input['Log_Response_Time'] = np.log1p(data.get('response_time', 0))
    
    # Placeholder for training-only columns
    if 'connected_handling_time' in df_input.columns:
        df_input['connected_handling_time'] = 0

    # 2. Map Categorical Features (One-Hot Encoding)
    mappings = [
        f"channel_name_{data['channel']}",
        f"category_{data['category']}",
        f"Sub-category_{data['sub_category']}",
        f"Tenure Bucket_{data['tenure']}"
    ]
    
    for key in mappings:
        if key in df_input.columns:
            df_input[key] = 1

    # 3. Scale and Return
    return scaler.transform(df_input)

# --- USER INTERFACE ---
st.title("🛡️ DeepCSAT Early Warning System")
st.markdown("Predict customer dissatisfaction risk using **XGBoost** and **Sentiment Analysis**.")



with st.form("deep_csat_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Contextual Data")
        channel = st.selectbox("Channel", ["Inbound", "Outcall", "Email"])
        category = st.selectbox("Category", [
            "Order Related", "Returns", "Cancellation", 
            "Refund Related", "Product Queries", "Feedback"
        ])
        sub_cat = st.text_input("Sub-Category", "General Enquiry")
        tenure = st.selectbox("Agent Tenure Bucket", ["0-30", "31-60", "61-90", ">90", "On Job Training"])
    
    with col2:
        st.subheader("Interaction Details")
        price = st.number_input("Item Price", min_value=0.0, value=50.0)
        resp_time = st.number_input("Response Time (Minutes)", min_value=0, value=15)
        remarks = st.text_area("Customer Remarks", height=100, placeholder="Paste feedback text here...")

    submit = st.form_submit_button("ANALYZE CSAT RISK")

# --- PREDICTION LOGIC ---
if submit:
    with st.spinner("Processing signals..."):
        input_data = {
            'channel': channel, 'category': category, 'sub_category': sub_cat,
            'tenure': tenure, 'Item_price': price, 'remarks': remarks,
            'response_time': resp_time
        }
        
        # Preprocess and Predict
        processed_x = preprocess_input(input_data, feature_names, scaler)
        prediction = model.predict(processed_x)[0]
        prob = model.predict_proba(processed_x)[0][1]

        st.divider()

        

        if prediction == 0:  # Assuming 0 is the 'High Risk/Low CSAT' class
            st.error("### 🚩 HIGH DISSATISFACTION RISK")
            st.metric("Risk Level", f"{prob:.1%}")
            st.warning("**Recommendation:** Proactively escalate to a Senior Manager immediately.")
        else:
            st.success("### ✅ LOW RISK")
            st.metric("Satisfaction Confidence", f"{1-prob:.1%}")
            st.write("The interaction data suggests a likely positive CSAT score.")

