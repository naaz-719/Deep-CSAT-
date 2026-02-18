"""
Flask Web Application for CSAT Score Prediction
Modern, immersive UI with Tailwind CSS
"""

from flask import Flask, render_template, request, jsonify, session
import numpy as np
import pandas as pd
import json
import pickle
from tensorflow import keras
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and preprocessing objects
MODEL_PATH = 'models/csat_model.keras'
SCALER_PATH = 'models/scaler.pkl'
ENCODERS_PATH = 'models/encoders.pkl'
FEATURE_COLUMNS_PATH = 'models/feature_columns.json'
PERFORMANCE_METRICS_PATH = 'models/performance_metrics.json'

# Global variables for loaded models
model = None
scaler = None
encoders = None
feature_columns = None
performance_metrics = None

# Real-time statistics tracking
prediction_stats = {
    'total_predictions': 0,
    'average_score': 0.0,
    'score_distribution': {
        'highly_satisfied': 0,    # 4.5-5.0
        'satisfied': 0,            # 3.5-4.5
        'neutral': 0,              # 2.5-3.5
        'dissatisfied': 0,         # 1.5-2.5
        'highly_dissatisfied': 0   # 1.0-1.5
    },
    'recent_predictions': [],  # Last 10 predictions
    'last_updated': None,
    'hourly_predictions': {},  # Track predictions by hour
    'daily_predictions': {},   # Track predictions by day
    'score_trend': [],  # Track score trend over time (all predictions)
    'satisfaction_rate': 0.0,  # Percentage of satisfied customers (>3.5)
    'min_score': None,
    'max_score': None,
    'std_deviation': 0.0,
    'prediction_times': [],  # Track inference times
    
    # Advanced analytics data
    'sentiment_data': {
        'positive': [],  # Track positive sentiment scores
        'negative': []   # Track negative sentiment scores
    },
    'feature_importance': {
        'response_time': 0,
        'sentiment': 0,
        'price': 0,
        'handling_time': 0,
        'agent_performance': 0
    },
    'segment_data': {
        'high_value': [],  # High price items
        'mid_value': [],   # Mid price items
        'low_value': [],   # Low price items
        'weekend': [],     # Weekend predictions
        'weekday': []      # Weekday predictions
    },
    'emotion_heatmap': {
        'morning': {'positive': 0, 'neutral': 0, 'negative': 0},
        'afternoon': {'positive': 0, 'neutral': 0, 'negative': 0},
        'evening': {'positive': 0, 'neutral': 0, 'negative': 0},
        'night': {'positive': 0, 'neutral': 0, 'negative': 0}
    },
    'feedback_topics': {
        'delivery': 0,
        'quality': 0,
        'price': 0,
        'service': 0,
        'support': 0
    },
    'service_metrics': {
        'response_times': [],  # Track response times
        'handling_times': [],  # Track handling times
        'scores': []           # Parallel array of scores
    },
    'anomalies': [],  # Track detected anomalies
    'risk_analysis': {
        'high_risk': [],   # Low scores with specific patterns
        'medium_risk': [], # Moderate scores
        'low_risk': []     # High scores
    },
    'real_time_stream': []  # Last 20 predictions for streaming display
}


def load_model_and_preprocessors():
    """Load trained model and preprocessing objects"""
    global model, scaler, encoders, feature_columns, performance_metrics
    
    try:
        # Load Keras model
        if os.path.exists(MODEL_PATH):
            model = keras.models.load_model(MODEL_PATH)
            print("✓ Model loaded successfully")
        
        # Load scaler
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("✓ Scaler loaded successfully")
        
        # Load encoders
        if os.path.exists(ENCODERS_PATH):
            with open(ENCODERS_PATH, 'rb') as f:
                encoders = pickle.load(f)
            print("✓ Encoders loaded successfully")
        
        # Load feature columns
        if os.path.exists(FEATURE_COLUMNS_PATH):
            with open(FEATURE_COLUMNS_PATH, 'r') as f:
                feature_columns = json.load(f)
            print("✓ Feature columns loaded successfully")
        
        # Load performance metrics
        if os.path.exists(PERFORMANCE_METRICS_PATH):
            with open(PERFORMANCE_METRICS_PATH, 'r') as f:
                performance_metrics = json.load(f)
            print("✓ Performance metrics loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading model/preprocessors: {str(e)}")
        return False


# Load model on startup
load_model_and_preprocessors()


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Home page with hero section and feature showcase"""
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    """Prediction form page"""
    # Pass feature columns to template for dynamic form generation
    return render_template('predict.html', feature_columns=feature_columns)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for CSAT score prediction
    Accepts JSON with customer interaction features
    """
    global prediction_stats
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Create feature array in correct order
        features = []
        for col in feature_columns:
            if col in data:
                features.append(float(data[col]))
            else:
                features.append(0.0)  # Default value for missing features
        
        # Convert to numpy array and reshape
        features_array = np.array([features])
        
        # Scale features
        if scaler is not None:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Make prediction
        if model is not None:
            prediction = model.predict(features_scaled, verbose=0)
            csat_score = float(prediction[0][0])
            
            # Clip score to valid range (1-5)
            csat_score = np.clip(csat_score, 1.0, 5.0)
            
            # Determine satisfaction level
            if csat_score >= 4.5:
                level = "Highly Satisfied"
                color = "green"
                category = 'highly_satisfied'
            elif csat_score >= 3.5:
                level = "Satisfied"
                color = "blue"
                category = 'satisfied'
            elif csat_score >= 2.5:
                level = "Neutral"
                color = "yellow"
                category = 'neutral'
            elif csat_score >= 1.5:
                level = "Dissatisfied"
                color = "orange"
                category = 'dissatisfied'
            else:
                level = "Highly Dissatisfied"
                color = "red"
                category = 'highly_dissatisfied'
            
            # Update real-time statistics
            prediction_stats['total_predictions'] += 1
            prediction_stats['score_distribution'][category] += 1
            
            # Update average score
            current_avg = prediction_stats['average_score']
            total = prediction_stats['total_predictions']
            prediction_stats['average_score'] = ((current_avg * (total - 1)) + csat_score) / total
            
            # Update min/max scores
            if prediction_stats['min_score'] is None or csat_score < prediction_stats['min_score']:
                prediction_stats['min_score'] = csat_score
            if prediction_stats['max_score'] is None or csat_score > prediction_stats['max_score']:
                prediction_stats['max_score'] = csat_score
            
            # Update score trend (keep all predictions)
            prediction_stats['score_trend'].append(round(csat_score, 2))
            
            # Calculate standard deviation
            if len(prediction_stats['score_trend']) > 1:
                prediction_stats['std_deviation'] = float(np.std(prediction_stats['score_trend']))
            
            # Calculate satisfaction rate (>3.5)
            satisfied_count = (prediction_stats['score_distribution']['highly_satisfied'] + 
                             prediction_stats['score_distribution']['satisfied'])
            prediction_stats['satisfaction_rate'] = (satisfied_count / total) * 100 if total > 0 else 0
            
            # Track hourly predictions
            current_hour = datetime.now().strftime('%Y-%m-%d %H:00')
            prediction_stats['hourly_predictions'][current_hour] = prediction_stats['hourly_predictions'].get(current_hour, 0) + 1
            
            # Track daily predictions
            current_day = datetime.now().strftime('%Y-%m-%d')
            prediction_stats['daily_predictions'][current_day] = prediction_stats['daily_predictions'].get(current_day, 0) + 1
            
            # Add to recent predictions (keep last 10)
            prediction_record = {
                'score': round(csat_score, 2),
                'level': level,
                'timestamp': datetime.now().isoformat(),
                'color': color
            }
            prediction_stats['recent_predictions'].insert(0, prediction_record)
            if len(prediction_stats['recent_predictions']) > 10:
                prediction_stats['recent_predictions'].pop()
            
            # Advanced Analytics Tracking
            
            # 1. Sentiment data tracking
            if 'positive_sentiment' in data:
                prediction_stats['sentiment_data']['positive'].append(float(data['positive_sentiment']))
                if len(prediction_stats['sentiment_data']['positive']) > 50:
                    prediction_stats['sentiment_data']['positive'].pop(0)
            
            if 'negative_sentiment' in data:
                prediction_stats['sentiment_data']['negative'].append(float(data['negative_sentiment']))
                if len(prediction_stats['sentiment_data']['negative']) > 50:
                    prediction_stats['sentiment_data']['negative'].pop(0)
            
            # 2. Feature importance (simulated correlation)
            if 'response_time_hours' in data:
                prediction_stats['feature_importance']['response_time'] += abs(5.0 - csat_score) * float(data.get('response_time_hours', 0)) / 24.0
            if 'positive_sentiment' in data:
                prediction_stats['feature_importance']['sentiment'] += csat_score * float(data.get('positive_sentiment', 0))
            if 'Item_price' in data:
                prediction_stats['feature_importance']['price'] += csat_score * float(data.get('Item_price', 0)) / 1000.0
            if 'connected_handling_time' in data:
                prediction_stats['feature_importance']['handling_time'] += abs(3.0 - csat_score) * float(data.get('connected_handling_time', 0)) / 60.0
            if 'agent_case_count' in data:
                prediction_stats['feature_importance']['agent_performance'] += csat_score * float(data.get('agent_case_count', 0)) / 200.0
            
            # 3. Segment data tracking
            item_price = float(data.get('Item_price', 0))
            if item_price > 1000:
                prediction_stats['segment_data']['high_value'].append(csat_score)
            elif item_price > 500:
                prediction_stats['segment_data']['mid_value'].append(csat_score)
            else:
                prediction_stats['segment_data']['low_value'].append(csat_score)
            
            # Keep segment data limited
            for segment in prediction_stats['segment_data']:
                if len(prediction_stats['segment_data'][segment]) > 30:
                    prediction_stats['segment_data'][segment].pop(0)
            
            # Weekend vs Weekday
            is_weekend = data.get('is_weekend', 0)
            if is_weekend == 1:
                prediction_stats['segment_data']['weekend'].append(csat_score)
            else:
                prediction_stats['segment_data']['weekday'].append(csat_score)
            
            # 4. Emotion heatmap (time of day + sentiment)
            current_hour = datetime.now().hour
            if current_hour < 6:
                time_period = 'night'
            elif current_hour < 12:
                time_period = 'morning'
            elif current_hour < 18:
                time_period = 'afternoon'
            else:
                time_period = 'evening'
            
            pos_sent = float(data.get('positive_sentiment', 0))
            neg_sent = float(data.get('negative_sentiment', 0))
            
            if pos_sent > 0.6:
                prediction_stats['emotion_heatmap'][time_period]['positive'] += 1
            elif neg_sent > 0.6:
                prediction_stats['emotion_heatmap'][time_period]['negative'] += 1
            else:
                prediction_stats['emotion_heatmap'][time_period]['neutral'] += 1
            
            # 5. Feedback topics (simulated based on features)
            if float(data.get('response_time_hours', 0)) > 5:
                prediction_stats['feedback_topics']['delivery'] += 1
            if item_price > 800:
                prediction_stats['feedback_topics']['quality'] += 1
            if item_price > 1500:
                prediction_stats['feedback_topics']['price'] += 1
            if float(data.get('connected_handling_time', 0)) > 20:
                prediction_stats['feedback_topics']['service'] += 1
            if csat_score < 3.0:
                prediction_stats['feedback_topics']['support'] += 1
            
            # 6. Service metrics tracking
            if 'response_time_hours' in data:
                prediction_stats['service_metrics']['response_times'].append(float(data['response_time_hours']))
            if 'connected_handling_time' in data:
                prediction_stats['service_metrics']['handling_times'].append(float(data['connected_handling_time']))
            prediction_stats['service_metrics']['scores'].append(csat_score)
            
            # Keep last 50
            for key in prediction_stats['service_metrics']:
                if len(prediction_stats['service_metrics'][key]) > 50:
                    prediction_stats['service_metrics'][key].pop(0)
            
            # 7. Anomaly detection (simple threshold-based)
            if len(prediction_stats['score_trend']) > 5:
                recent_avg = np.mean(prediction_stats['score_trend'][-5:])
                if abs(csat_score - recent_avg) > 1.5:  # Significant deviation
                    prediction_stats['anomalies'].append({
                        'index': len(prediction_stats['score_trend']) - 1,
                        'score': csat_score,
                        'timestamp': datetime.now().isoformat()
                    })
                    if len(prediction_stats['anomalies']) > 10:
                        prediction_stats['anomalies'].pop(0)
            
            # 8. Risk analysis
            response_time = float(data.get('response_time_hours', 0))
            neg_sent = float(data.get('negative_sentiment', 0))
            
            risk_score = (5.0 - csat_score) * 0.4 + response_time * 0.3 + neg_sent * 0.3
            
            risk_data = {
                'score': csat_score,
                'risk': risk_score,
                'timestamp': datetime.now().isoformat()
            }
            
            if risk_score > 2.5:
                prediction_stats['risk_analysis']['high_risk'].append(risk_data)
            elif risk_score > 1.5:
                prediction_stats['risk_analysis']['medium_risk'].append(risk_data)
            else:
                prediction_stats['risk_analysis']['low_risk'].append(risk_data)
            
            # Keep last 30 for each risk level
            for risk_level in prediction_stats['risk_analysis']:
                if len(prediction_stats['risk_analysis'][risk_level]) > 30:
                    prediction_stats['risk_analysis'][risk_level].pop(0)
            
            # 9. Real-time stream (keep last 20)
            stream_record = {
                'score': round(csat_score, 2),
                'level': level,
                'emoji': '😊' if csat_score > 4 else '🙂' if csat_score > 3 else '😐' if csat_score > 2 else '😟',
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'color': color
            }
            prediction_stats['real_time_stream'].insert(0, stream_record)
            if len(prediction_stats['real_time_stream']) > 20:
                prediction_stats['real_time_stream'].pop()
            
            prediction_stats['last_updated'] = datetime.now().isoformat()
            
            # Return prediction result
            return jsonify({
                'success': True,
                'csat_score': round(csat_score, 2),
                'satisfaction_level': level,
                'color': color,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Dashboard page with model performance metrics"""
    if performance_metrics is None:
        metrics = {
            'mse': 'N/A',
            'rmse': 'N/A',
            'mae': 'N/A',
            'r2': 'N/A'
        }
    else:
        metrics = performance_metrics
    
    return render_template('dashboard.html', metrics=metrics)


@app.route('/dashboard/advanced')
def dashboard_advanced():
    """Advanced Analytics Dashboard with 10 comprehensive visualizations"""
    return render_template('dashboard_advanced.html')


@app.route('/api/model-info')
def api_model_info():
    """API endpoint to get model information"""
    try:
        info = {
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None,
            'encoders_loaded': encoders is not None,
            'feature_count': len(feature_columns) if feature_columns else 0,
            'feature_columns': feature_columns,
            'performance_metrics': performance_metrics
        }
        
        if model is not None:
            info['model_summary'] = {
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'total_params': model.count_params()
            }
        
        return jsonify(info)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def api_stats():
    """API endpoint to get real-time prediction statistics"""
    try:
        # Calculate percentages for distribution
        total = prediction_stats['total_predictions']
        distribution_percentages = {}
        
        if total > 0:
            for category, count in prediction_stats['score_distribution'].items():
                distribution_percentages[category] = round((count / total) * 100, 1)
        else:
            distribution_percentages = {k: 0 for k in prediction_stats['score_distribution'].keys()}
        
        # Get last 24 hours of predictions
        hourly_data = dict(list(prediction_stats['hourly_predictions'].items())[-24:])
        
        # Get last 7 days of predictions
        daily_data = dict(list(prediction_stats['daily_predictions'].items())[-7:])
        
        # Calculate segment averages
        segment_averages = {}
        for segment, scores in prediction_stats['segment_data'].items():
            if scores:
                segment_averages[segment] = round(np.mean(scores), 2)
            else:
                segment_averages[segment] = 0
        
        # Normalize feature importance
        total_importance = sum(prediction_stats['feature_importance'].values())
        feature_importance_norm = {}
        if total_importance > 0:
            for feature, value in prediction_stats['feature_importance'].items():
                feature_importance_norm[feature] = round((value / total_importance) * 100, 1)
        else:
            feature_importance_norm = {k: 0 for k in prediction_stats['feature_importance'].keys()}
        
        return jsonify({
            'success': True,
            'total_predictions': prediction_stats['total_predictions'],
            'average_score': round(prediction_stats['average_score'], 2),
            'min_score': round(prediction_stats['min_score'], 2) if prediction_stats['min_score'] else 0,
            'max_score': round(prediction_stats['max_score'], 2) if prediction_stats['max_score'] else 0,
            'std_deviation': round(prediction_stats['std_deviation'], 3),
            'satisfaction_rate': round(prediction_stats['satisfaction_rate'], 1),
            'score_distribution': prediction_stats['score_distribution'],
            'distribution_percentages': distribution_percentages,
            'recent_predictions': prediction_stats['recent_predictions'],
            'score_trend': prediction_stats['score_trend'],
            'hourly_predictions': hourly_data,
            'daily_predictions': daily_data,
            
            # Advanced analytics data
            'sentiment_data': prediction_stats['sentiment_data'],
            'feature_importance': feature_importance_norm,
            'segment_averages': segment_averages,
            'emotion_heatmap': prediction_stats['emotion_heatmap'],
            'feedback_topics': prediction_stats['feedback_topics'],
            'service_metrics': prediction_stats['service_metrics'],
            'anomalies': prediction_stats['anomalies'],
            'risk_analysis': prediction_stats['risk_analysis'],
            'real_time_stream': prediction_stats['real_time_stream'],
            
            'last_updated': prediction_stats['last_updated']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):
    """Custom 404 error page"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Custom 500 error page"""
    return render_template('500.html'), 500


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting CSAT Prediction Web Application")
    print("="*60)
    print(f"📊 Model Status: {'✓ Loaded' if model else '✗ Not Loaded'}")
    print(f"📐 Scaler Status: {'✓ Loaded' if scaler else '✗ Not Loaded'}")
    print(f"🔧 Features: {len(feature_columns) if feature_columns else 0}")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
