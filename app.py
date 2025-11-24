import joblib
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import datetime as dt
from sklearn import set_config
from time import sleep
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Try importing visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import shap
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-v0_8-darkgrid')
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

set_config(transform_output="pandas")

# Page configuration
st.set_page_config(
    page_title="Uber Demand Forecasting Platform",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - FIXED FONT SIZES
st.markdown("""
    <style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Fixed header sizes */
    .main-header {
        font-size: 2rem !important;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.2rem !important;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
        font-weight: 500;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card h3 {
        font-size: 1.8rem !important;
        margin: 0.5rem 0;
        font-weight: 700;
    }
    
    .metric-card p {
        font-size: 0.9rem !important;
        margin: 0;
        opacity: 0.9;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 0.8rem;
        margin: 0.8rem 0;
        border-radius: 4px;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.8rem;
        margin: 0.8rem 0;
        border-radius: 4px;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 0.8rem;
        margin: 0.8rem 0;
        border-radius: 4px;
    }
    
    /* Insights section */
    .insight-card {
        background-color: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .insight-card h4 {
        font-size: 1rem !important;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .insight-card p {
        font-size: 0.9rem !important;
        color: #555;
        margin: 0;
    }
    
    /* Region cards */
    .region-card {
        background-color: white;
        border-radius: 6px;
        padding: 0.6rem;
        margin: 0.4rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        color: #666;
        font-size: 0.85rem !important;
    }
    
    /* Fix streamlit default font sizes */
    .stMarkdown {
        font-size: 0.95rem !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Set the root path
root_path = Path(__file__).parent

# File paths
plot_data_path = root_path / "data/external/plot_data.csv"
data_path = root_path / "data/processed/test.csv"
train_data_path = root_path / "data/processed/train.csv"
kmeans_path = root_path / "models/mb_kmeans.joblib"
scaler_path = root_path / "models/scaler.joblib"
encoder_path = root_path / "models/encoder.joblib"
model_path = root_path / "models/model.keras"
metadata_path = root_path / "models/model_metadata.json"

# ============================================
# UTILITY FUNCTIONS
# ============================================
def calculate_prediction_confidence(predictions, actuals):
    """Calculate confidence metrics for predictions"""
    errors = np.abs(predictions - actuals)
    confidence_scores = 100 - (errors / actuals * 100)
    confidence_scores = np.clip(confidence_scores, 0, 100)
    return confidence_scores

def get_time_of_day_category(hour):
    """Categorize time of day"""
    if 5 <= hour < 9:
        return "Morning Rush (5am-9am)"
    elif 9 <= hour < 12:
        return "Mid-Morning (9am-12pm)"
    elif 12 <= hour < 17:
        return "Afternoon (12pm-5pm)"
    elif 17 <= hour < 21:
        return "Evening Rush (5pm-9pm)"
    elif 21 <= hour < 24:
        return "Late Night (9pm-12am)"
    else:
        return "Overnight (12am-5am)"

def get_demand_category(demand):
    """Categorize demand level"""
    if demand < 20:
        return "🟢 Low", "low"
    elif demand < 50:
        return "🟡 Medium", "medium"
    elif demand < 80:
        return "🟠 High", "high"
    else:
        return "🔴 Very High", "very_high"

def analyze_prediction_quality(predictions, actuals):
    """Detailed analysis of prediction quality"""
    errors = predictions - actuals
    abs_errors = np.abs(errors)
    
    analysis = {
        'mae': np.mean(abs_errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'mape': np.mean(abs_errors / actuals) * 100,
        'overestimate_pct': (errors > 0).sum() / len(errors) * 100,
        'underestimate_pct': (errors < 0).sum() / len(errors) * 100,
        'within_10pct': (abs_errors / actuals <= 0.1).sum() / len(errors) * 100,
        'within_20pct': (abs_errors / actuals <= 0.2).sum() / len(errors) * 100,
        'max_error': abs_errors.max(),
        'avg_confidence': calculate_prediction_confidence(predictions, actuals).mean()
    }
    
    return analysis

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    """Load all models with error handling"""
    try:
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        model = keras.models.load_model(model_path)
        kmeans = joblib.load(kmeans_path)
        
        # Load metadata
        metadata = None
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return scaler, encoder, model, kmeans, metadata, None
    except Exception as e:
        return None, None, None, None, None, str(e)

@st.cache_data
def load_data():
    """Load datasets with error handling"""
    try:
        df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")
        
        df_train = None
        if train_data_path.exists():
            df_train = pd.read_csv(train_data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")
        
        if plot_data_path.exists():
            df_plot = pd.read_csv(plot_data_path)
        else:
            df_plot = create_plot_data_from_test(df)
        
        return df_plot, df, df_train, None
    except Exception as e:
        return None, None, None, str(e)

def create_plot_data_from_test(df_test):
    """Create plot_data if it doesn't exist"""
    regions = df_test['region'].unique() if 'region' in df_test.columns else range(30)
    
    lat_range = (40.60, 40.85)
    lon_range = (-74.05, -73.70)
    
    plot_data = []
    for region in regions:
        n_points = 100
        lats = np.random.uniform(lat_range[0], lat_range[1], n_points)
        lons = np.random.uniform(lon_range[0], lon_range[1], n_points)
        
        for lat, lon in zip(lats, lons):
            plot_data.append({
                'pickup_latitude': lat,
                'pickup_longitude': lon,
                'region': int(region)
            })
    
    return pd.DataFrame(plot_data)

# Load models and data
with st.spinner("🔄 Loading models and data..."):
    scaler, encoder, model, kmeans, metadata, model_error = load_models()
    
    if model_error:
        st.error(f"❌ Error loading models: {model_error}")
        st.stop()
    
    df_plot, df, df_train, data_error = load_data()
    
    if data_error:
        st.error(f"❌ Error loading data: {data_error}")
        st.stop()

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_with_gru(input_data, encoder, model):
    """Make predictions with GRU model"""
    X_encoded = encoder.transform(input_data)
    
    if isinstance(X_encoded, pd.DataFrame):
        X_encoded = X_encoded.values
    
    total_features = X_encoded.shape[1]
    X_reshaped = X_encoded.reshape(X_encoded.shape[0], 1, total_features)
    
    predictions = model.predict(X_reshaped, verbose=0).flatten()
    
    return predictions, X_reshaped, X_encoded

# ============================================
# SHAP FUNCTIONS (FIXED)
# ============================================
if SHAP_AVAILABLE:
    @st.cache_resource
    def create_shap_explainer(_model, _encoder, background_data_raw, max_samples=50):
        """Create SHAP KernelExplainer"""
        try:
            def predict_fn(X_raw):
                X_encoded = _encoder.transform(pd.DataFrame(X_raw, columns=background_data_raw.columns))
                
                if isinstance(X_encoded, pd.DataFrame):
                    X_encoded = X_encoded.values
                
                total_features = X_encoded.shape[1]
                X_reshaped = X_encoded.reshape(X_encoded.shape[0], 1, total_features)
                
                predictions = _model.predict(X_reshaped, verbose=0).flatten()
                return predictions
            
            background_sample = background_data_raw.sample(min(max_samples, len(background_data_raw)))
            explainer = shap.KernelExplainer(predict_fn, background_sample)
            
            return explainer, predict_fn
        except Exception as e:
            st.error(f"Error creating SHAP explainer: {e}")
            return None, None

    def get_feature_importance(shap_values, feature_names, top_n=10):
        """Calculate feature importance"""
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(mean_abs_shap)],
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df

# ============================================
# UI - HEADER
# ============================================
st.markdown('<div class="main-header">🚕 Uber Demand Forecasting Platform 🌆</div>', unsafe_allow_html=True)

# Show model info
if metadata:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", metadata['model_name'])
    with col2:
        st.metric("Training MAE", f"{metadata['final_train_mae']:.2f}")
    with col3:
        st.metric("Validation MAE", f"{metadata['final_val_mae']:.2f}")
    with col4:
        st.metric("Epochs Trained", metadata['epochs_trained'])

# ============================================
# SIDEBAR
# ============================================
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

map_type = st.sidebar.radio(
    "📍 Map View",
    ["Complete NYC Map", "Neighborhood View"],
    index=1
)

show_advanced = st.sidebar.checkbox("🔬 Advanced Analysis", value=False)
show_shap = st.sidebar.checkbox("🧠 SHAP Explanations", value=False) if SHAP_AVAILABLE else False

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About")
st.sidebar.info("""
**Features:**
- Real-time demand prediction
- Interactive heatmaps
- Confidence scoring
- Temporal analysis
- SHAP explanations
- Performance metrics
""")

# ============================================
# MAIN TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Prediction", 
    "📊 Analysis", 
    "🔬 Explainability" if SHAP_AVAILABLE else "📈 Insights",
    "📈 Performance"
])

# ============================================
# TAB 1: PREDICTION
# ============================================
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📅 Select Date")
        date = st.date_input(
            "Date",
            value=None,
            min_value=dt.date(year=2016, month=3, day=1),
            max_value=dt.date(year=2016, month=3, day=31),
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("### ⏰ Select Time")
        time = st.time_input("Time", value=None, label_visibility="collapsed")
    
    if date and time:
        delta = dt.timedelta(minutes=15)
        next_interval = dt.datetime(
            year=date.year,
            month=date.month,
            day=date.day,
            hour=time.hour,
            minute=time.minute
        ) + delta
        
        # Time analysis
        time_category = get_time_of_day_category(next_interval.hour)
        day_name = next_interval.strftime("%A")
        
        st.markdown(f"""
        <div class="info-box">
            <b>Prediction Time:</b> {next_interval.strftime('%I:%M %p')} | 
            <b>Day:</b> {day_name} | 
            <b>Period:</b> {time_category}
        </div>
        """, unsafe_allow_html=True)
        
        index = pd.Timestamp(f"{date} {next_interval.time()}")
        
        # Location
        sample_loc = df_plot.sample(1).reset_index(drop=True)
        lat = sample_loc["pickup_latitude"].item()
        long = sample_loc["pickup_longitude"].item()
        region = sample_loc["region"].item()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📍 Latitude", f"{lat:.4f}")
        with col2:
            st.metric("📍 Longitude", f"{long:.4f}")
        with col3:
            st.metric("🗺️ Region ID", region)
        
        scaled_cord = scaler.transform(sample_loc.iloc[:, 0:2])
        
        # MAP
        st.markdown("### 🗺️ NYC Demand Heatmap")
        
        colors = [
            "#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F",
            "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00",
            "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF",
            "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3",
            "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347",
            "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"
        ]
        
        region_colors = {r: colors[i] for i, r in enumerate(df_plot["region"].unique().tolist())}
        df_plot["color"] = df_plot["region"].map(region_colors)
        
        if map_type == "Complete NYC Map":
            with st.spinner("Loading map..."):
                st.map(df_plot, latitude="pickup_latitude", longitude="pickup_longitude", size=8, color="color")
            
            input_data = df.loc[index, :].sort_values("region")
            target = input_data["total_pickups"]
            
            predictions, X_reshaped, X_encoded = predict_with_gru(
                input_data.drop(columns=["total_pickups"]),
                encoder,
                model
            )
            
            # Store session state
            st.session_state.update({
                'predictions': predictions,
                'X_reshaped': X_reshaped,
                'input_data_raw': input_data.drop(columns=["total_pickups"]),
                'current_region': region,
                'actuals': target.values,
                'timestamp': next_interval,
                'time_category': time_category,
                'day_name': day_name
            })
            
            # Analysis
            analysis = analyze_prediction_quality(predictions, target.values)
            confidence_scores = calculate_prediction_confidence(predictions, target.values)
            
            # Summary Metrics
            st.markdown("### 📊 Prediction Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p>Avg Predicted</p>
                    <h3>{int(predictions.mean())}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <p>Avg Actual</p>
                    <h3>{int(target.mean())}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <p>MAE</p>
                    <h3>{analysis['mae']:.1f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <p>Avg Confidence</p>
                    <h3>{analysis['avg_confidence']:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Top regions
            st.markdown("### 🏆 Top 10 High Demand Regions")
            
            top_10_idx = np.argsort(predictions)[-10:][::-1]
            
            for idx in top_10_idx:
                pred = predictions[idx]
                actual = target.iloc[idx]
                conf = confidence_scores[idx]
                demand_label, demand_cat = get_demand_category(pred)
                
                col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color:{colors[idx]}; width:40px; height:40px; 
                    border-radius:50%; margin:auto;"></div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    is_current = "📍 " if idx == region else ""
                    st.markdown(f"**{is_current}Region {idx}** - {demand_label}")
                
                with col3:
                    st.metric("Predicted", int(pred), delta=f"{int(pred-actual):+d}")
                
                with col4:
                    st.metric("Confidence", f"{conf:.0f}%")
            
        else:  # Neighborhood View
            distances = kmeans.transform(scaled_cord).values.ravel().tolist()
            distances = list(enumerate(distances))
            sorted_distances = sorted(distances, key=lambda x: x[1])[0:9]
            indexes = sorted([ind[0] for ind in sorted_distances])
            
            df_plot_filtered = df_plot[df_plot["region"].isin(indexes)]
            
            with st.spinner("Loading neighborhood map..."):
                st.map(df_plot_filtered, latitude="pickup_latitude", longitude="pickup_longitude", size=10, color="color")
            
            input_data = df.loc[index, :]
            input_data = input_data.loc[input_data["region"].isin(indexes), :].sort_values("region")
            target = input_data["total_pickups"]
            
            predictions, X_reshaped, X_encoded = predict_with_gru(
                input_data.drop(columns=["total_pickups"]),
                encoder,
                model
            )
            
            st.session_state.update({
                'predictions': predictions,
                'X_reshaped': X_reshaped,
                'input_data_raw': input_data.drop(columns=["total_pickups"]),
                'current_region': region,
                'actuals': target.values,
                'indexes': indexes,
                'timestamp': next_interval,
                'time_category': time_category,
                'day_name': day_name
            })
            
            # Display regions
            st.markdown("### 📍 Nearby Regions")
            
            for i, idx in enumerate(indexes):
                pred = predictions[i]
                actual = target.iloc[i]
                demand_label, _ = get_demand_category(pred)
                conf = calculate_prediction_confidence(np.array([pred]), np.array([actual]))[0]
                
                col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
                
                with col1:
                    st.markdown(f"""
                    <div style="background-color:{colors[idx]}; width:40px; height:40px; 
                    border-radius:50%; margin:auto;"></div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    is_current = "📍 " if idx == region else ""
                    st.markdown(f"**{is_current}Region {idx}** - {demand_label}")
                
                with col3:
                    st.metric("Predicted", int(pred), delta=f"{int(pred-actual):+d}")
                
                with col4:
                    st.metric("Confidence", f"{conf:.0f}%")

# ============================================
# TAB 2: ANALYSIS
# ============================================
with tab2:
    if 'predictions' in st.session_state:
        st.markdown("### 📊 Detailed Analysis")
        
        predictions = st.session_state['predictions']
        actuals = st.session_state['actuals']
        analysis = analyze_prediction_quality(predictions, actuals)
        
        # Key Insights
        st.markdown("#### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="insight-card">
                <h4>🎯 Prediction Accuracy</h4>
                <p><b>{analysis['within_10pct']:.1f}%</b> of predictions within 10% error</p>
                <p><b>{analysis['within_20pct']:.1f}%</b> within 20% error</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>📈 Prediction Bias</h4>
                <p><b>{analysis['overestimate_pct']:.1f}%</b> overestimations</p>
                <p><b>{analysis['underestimate_pct']:.1f}%</b> underestimations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-card">
                <h4>📏 Error Metrics</h4>
                <p><b>MAE:</b> {analysis['mae']:.2f} rides</p>
                <p><b>RMSE:</b> {analysis['rmse']:.2f} rides</p>
                <p><b>MAPE:</b> {analysis['mape']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>🎲 Confidence</h4>
                <p><b>Average:</b> {analysis['avg_confidence']:.1f}%</p>
                <p><b>Max Error:</b> {analysis['max_error']:.0f} rides</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        if PLOTLY_AVAILABLE:
            st.markdown("#### 📈 Visualizations")
            
            # Prediction vs Actual
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Predicted vs Actual', 'Error Distribution')
            )
            
            # Scatter plot
            fig.add_trace(
                go.Scatter(
                    x=actuals,
                    y=predictions,
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=8, color='blue', opacity=0.6)
                ),
                row=1, col=1
            )
            
            # Perfect line
            min_val = min(actuals.min(), predictions.min())
            max_val = max(actuals.max(), predictions.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect',
                    line=dict(dash='dash', color='red')
                ),
                row=1, col=1
            )
            
            # Error distribution
            errors = predictions - actuals
            fig.add_trace(
                go.Histogram(
                    x=errors,
                    name='Errors',
                    marker_color='orange',
                    nbinsx=20
                ),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Actual Demand", row=1, col=1)
            fig.update_yaxes(title_text="Predicted Demand", row=1, col=1)
            fig.update_xaxes(title_text="Prediction Error", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Demand distribution
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=predictions,
                name='Predicted',
                opacity=0.7,
                marker_color='blue'
            ))
            fig2.add_trace(go.Histogram(
                x=actuals,
                name='Actual',
                opacity=0.7,
                marker_color='red'
            ))
            fig2.update_layout(
                title="Demand Distribution Comparison",
                xaxis_title="Demand",
                yaxis_title="Count",
                barmode='overlay',
                height=350
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Recommendations
        st.markdown("#### 🎯 Recommendations")
        
        if analysis['mape'] < 15:
            st.markdown("""
            <div class="success-box">
                ✅ <b>Excellent Prediction Quality</b><br>
                MAPE < 15% indicates high-quality predictions. The model is performing well.
            </div>
            """, unsafe_allow_html=True)
        elif analysis['mape'] < 25:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <b>Good Prediction Quality</b><br>
                MAPE between 15-25%. Consider model retraining with more recent data.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="warning-box">
                ⚠️ <b>Moderate Prediction Quality</b><br>
                MAPE > 25%. Model retraining recommended. Check for data drift.
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("📊 Make a prediction first to see detailed analysis.")

# ============================================
# TAB 3: EXPLAINABILITY
# ============================================
with tab3:
    if SHAP_AVAILABLE and show_shap:
        st.markdown("### 🔬 SHAP Explainability")
        
        if 'input_data_raw' in st.session_state and df_train is not None:
            st.info("⏱️ SHAP analysis takes 30-60 seconds. Click the button below to start.")
            
            if st.button("🚀 Compute SHAP Explanations", type="primary"):
                with st.spinner("Computing SHAP values... Please wait..."):
                    input_data_raw = st.session_state['input_data_raw']
                    feature_names = input_data_raw.columns.tolist()
                    
                    background_data = df_train.drop(columns=['total_pickups']).sample(min(50, len(df_train)))
                    
                    explainer, predict_fn = create_shap_explainer(model, encoder, background_data, max_samples=50)
                    
                    if explainer:
                        shap_values = explainer.shap_values(input_data_raw.head(5), nsamples=50)
                        
                        if isinstance(shap_values, list):
                            shap_values = shap_values[0]
                        
                        st.success("✅ SHAP analysis complete!")
                        
                        # Feature importance
                        importance_df = get_feature_importance(shap_values, feature_names, top_n=10)
                        
                        if PLOTLY_AVAILABLE:
                            fig = px.bar(
                                importance_df,
                                x='importance',
                                y='feature',
                                orientation='h',
                                title='Top 10 Most Important Features',
                                color='importance',
                                color_continuous_scale='Viridis'
                            )
                            fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Interpretation
                        st.markdown("#### 💡 Feature Interpretation")
                        
                        for idx, row in importance_df.head(5).iterrows():
                            feature_name = row['feature']
                            impact = row['importance']
                            
                            if 'lag' in feature_name.lower():
                                st.markdown(f"""
                                <div class="insight-card">
                                    <h4>🕐 {feature_name}</h4>
                                    <p><b>Type:</b> Historical Demand Pattern</p>
                                    <p><b>Impact Score:</b> {impact:.4f}</p>
                                    <p><b>Explanation:</b> Past demand at this time interval strongly influences current predictions.</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif 'day' in feature_name.lower() or 'week' in feature_name.lower():
                                st.markdown(f"""
                                <div class="insight-card">
                                    <h4>📅 {feature_name}</h4>
                                    <p><b>Type:</b> Temporal Pattern</p>
                                    <p><b>Impact Score:</b> {impact:.4f}</p>
                                    <p><b>Explanation:</b> Day of week affects demand patterns (weekday vs weekend).</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif 'region' in feature_name.lower():
                                st.markdown(f"""
                                <div class="insight-card">
                                    <h4>🗺️ {feature_name}</h4>
                                    <p><b>Type:</b> Geographic Factor</p>
                                    <p><b>Impact Score:</b> {impact:.4f}</p>
                                    <p><b>Explanation:</b> Location-based demand patterns vary across regions.</p>
                                </div>
                                """, unsafe_allow_html=True)
        else:
            st.warning("Make a prediction first and ensure training data is available.")
    else:
        st.markdown("### 📈 Model Insights")
        st.markdown("""
        <div class="info-box">
            <h4>🧠 How the Model Works</h4>
            <p><b>Architecture:</b> GRU (Gated Recurrent Unit) Neural Network</p>
            <p><b>Input Features:</b> 96 lag features (24 hours), region, day of week</p>
            <p><b>Prediction:</b> Demand for next 15-minute interval</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'input_data_raw' in st.session_state:
            input_data = st.session_state['input_data_raw']
            
            st.markdown("#### 📊 Feature Analysis")
            
            # Analyze lag features
            lag_cols = [col for col in input_data.columns if 'lag' in col.lower()]
            if lag_cols:
                first_row = input_data.iloc[0]
                lag_values = first_row[lag_cols[:24]].values
                
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=list(range(len(lag_values), 0, -1)),
                        y=lag_values,
                        mode='lines+markers',
                        name='Historical Demand',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=6)
                    ))
                    fig.update_layout(
                        title="Last 6 Hours of Demand Pattern (15-min intervals)",
                        xaxis_title="Periods Ago",
                        yaxis_title="Demand",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Insights
                recent_trend = "increasing" if lag_values[0] > lag_values[5] else "decreasing"
                avg_demand = lag_values.mean()
                
                st.markdown(f"""
                <div class="info-box">
                    <b>Recent Trend:</b> Demand is {recent_trend}<br>
                    <b>Average (last 6h):</b> {avg_demand:.1f} rides<br>
                    <b>Current Value:</b> {lag_values[0]:.0f} rides
                </div>
                """, unsafe_allow_html=True)

# ============================================
# TAB 4: PERFORMANCE
# ============================================
with tab4:
    st.markdown("### 📈 Model Performance Metrics")
    
    if metadata:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Training Performance")
            st.metric("Training Loss (MSE)", f"{metadata['final_train_loss']:.2f}")
            st.metric("Training MAE", f"{metadata['final_train_mae']:.2f}")
            st.metric("Epochs Trained", metadata['epochs_trained'])
        
        with col2:
            st.markdown("#### ✅ Validation Performance")
            st.metric("Validation Loss (MSE)", f"{metadata['final_val_loss']:.2f}")
            st.metric("Validation MAE", f"{metadata['final_val_mae']:.2f}")
            
            # Calculate improvement
            if 'predictions' in st.session_state:
                current_mae = analyze_prediction_quality(
                    st.session_state['predictions'],
                    st.session_state['actuals']
                )['mae']
                st.metric("Current Test MAE", f"{current_mae:.2f}")
    
    if 'predictions' in st.session_state:
        st.markdown("---")
        st.markdown("#### 📊 Current Session Metrics")
        
        analysis = analyze_prediction_quality(
            st.session_state['predictions'],
            st.session_state['actuals']
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MAE", f"{analysis['mae']:.2f}")
            st.metric("RMSE", f"{analysis['rmse']:.2f}")
        
        with col2:
            st.metric("MAPE", f"{analysis['mape']:.2f}%")
            st.metric("Max Error", f"{analysis['max_error']:.0f}")
        
        with col3:
            st.metric("Within 10%", f"{analysis['within_10pct']:.1f}%")
            st.metric("Avg Confidence", f"{analysis['avg_confidence']:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <p><b>🚕 Uber Demand Forecasting Platform</b></p>
        <p>Powered by GRU Neural Networks | SHAP Explainability | Real-time Analytics</p>
        <p style="font-size:0.75rem; margin-top:0.5rem;">
            Model Architecture: GRU | Framework: TensorFlow/Keras | XAI: SHAP KernelExplainer
        </p>
    </div>
""", unsafe_allow_html=True)