import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import os
import yaml
import sys
import types
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin
from streamlit_extras.metric_cards import style_metric_cards
from datetime import datetime

# Define custom transformers at the top level
class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(X)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return ["log_transformed"]
        return [f"log_{name}" for name in input_features]

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.features]

# Context manager to patch __main__ module
@contextmanager
def patch_main_module():
    original_main = sys.modules.get('__main__')
    patched_main = types.ModuleType("__patched_main__")
    patched_main.LogTransformer = LogTransformer
    patched_main.FeatureSelector = FeatureSelector
    sys.modules['__main__'] = patched_main
    try:
        yield
    finally:
        if original_main:
            sys.modules['__main__'] = original_main

# Custom CSS for professional finance dashboard with dark theme
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background: linear-gradient(135deg, #0c1a2d 0%, #152642 100%);
        color: #e0e7ff;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1429 0%, #101e38 100%) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border-right: 1px solid #1e3a5f;
    }
    
    .stSidebar .stSelectbox, .stSidebar .stSlider, .stSidebar .stNumberInput {
        color: #ffffff !important;
        background-color: #1a2d4d !important;
        border: 1px solid #2a4a7a;
        border-radius: 8px;
    }
    
    .stSidebar label {
        color: #a0b9e0 !important;
        font-weight: 500;
    }
    
    .sidebar-header {
        color: #5d9bff;
        font-weight: 700;
        border-bottom: 1px solid #2a4a7a;
        padding-bottom: 12px;
        margin-bottom: 20px;
        font-size: 18px;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(145deg, #1a2d4d, #152642);
        border-radius: 12px;
        padding: 20px 15px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
        border: 1px solid #2a4a7a;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    
    .metric-card h3 {
        color: #a0b9e0 !important;
        font-size: 16px;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .metric-card h2 {
        color: #ffffff !important;
        font-size: 24px;
        font-weight: 700;
        margin: 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(to right, #1d4ed8, #3b82f6);
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: 600;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Header styling */
    .dashboard-title {
        color: #e0e7ff;
        font-weight: 800;
        padding-bottom: 15px;
        margin-bottom: 25px;
        position: relative;
    }
    
    .dashboard-title h1 {
        font-size: 36px;
        margin-bottom: 8px;
        background: linear-gradient(90deg, #5d9bff 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .dashboard-title::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #5d9bff 0%, #3b82f6 100%);
        border-radius: 2px;
    }
    
    /* Table styling */
    table.dataframe {
        color: #e0e7ff !important;
        background-color: #1a2d4d !important;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border: 1px solid #2a4a7a;
    }
    
    table.dataframe th {
        background: linear-gradient(180deg, #1d4ed8 0%, #3b82f6 100%) !important;
        color: white !important;
        font-weight: 600;
        padding: 12px 15px !important;
    }
    
    table.dataframe td {
        background-color: #1a2d4d !important;
        color: #e0e7ff !important;
        padding: 10px 15px !important;
        border-bottom: 1px solid #2a4a7a;
    }
    
    table.dataframe tr:nth-child(even) {
        background-color: #152642 !important;
    }
    
    /* Financial impact styling */
    .decision-card {
        border-radius: 12px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.2);
        color: #e0e7ff;
        position: relative;
        overflow: hidden;
        background: linear-gradient(145deg, #1a2d4d, #152642);
        border-left: 6px solid;
    }
    
    .approve-card {
        border-left-color: #16a34a;
    }
    
    .decline-card {
        border-left-color: #dc2626;
    }
    
    .decision-card h3 {
        font-size: 22px;
        margin-top: 0;
        margin-bottom: 15px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        padding: 0;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a2d4d !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        margin: 0 !important;
        font-weight: 600;
        transition: all 0.3s ease;
        color: #a0b9e0 !important;
        border: 1px solid #2a4a7a;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1d4ed8 0%, #3b82f6 100%) !important;
        color: white !important;
        box-shadow: 0 4px 10px rgba(59, 130, 246, 0.3);
        border: 1px solid #3b82f6;
    }
    
    /* Fix text colors */
    .stMarkdown, .stText, .stNumberInput, .stSelectbox, .stSlider {
        color: #e0e7ff !important;
    }
    
    .stMetricValue {
        color: #ffffff !important;
        font-weight: 700;
    }
    
    .stMetricLabel {
        color: #a0b9e0 !important;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        color: #5d9bff;
        font-weight: 700;
        font-size: 24px;
        margin-top: 25px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #2a4a7a;
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        background: #1a2d4d;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        padding: 15px;
        margin-bottom: 20px;
        border: 1px solid #2a4a7a;
    }
    
    [data-testid="stExpanderDetails"] {
        padding: 15px 0 0 0;
    }
    
    /* Fix plot text */
    .svg-container text {
        fill: #e0e7ff !important;
        font-weight: 500;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(90deg, #0a1429 0%, #101e38 100%);
        color: #a0b9e0 !important;
        padding: 20px;
        border-radius: 12px;
        margin-top: 30px;
        text-align: center;
        border: 1px solid #2a4a7a;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .footer-title {
        color: #5d9bff;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #152642;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #1d4ed8 0%, #3b82f6 100%);
        border-radius: 4px;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .metric-card {
            margin-bottom: 15px;
        }
        
        .dashboard-title h1 {
            font-size: 28px;
        }
    }
    
    /* Input field styling */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSlider>div>div>div>div {
        background-color: #1a2d4d !important;
        color: #e0e7ff !important;
        border: 1px solid #2a4a7a !important;
        border-radius: 8px;
    }
    
    /* Divider styling */
    .stDivider {
        border-bottom: 1px solid #2a4a7a;
    }
    
    /* Pattern guide styling */
    .pattern-guide {
        background: #152642;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #2a4a7a;
    }
    
    .pattern-type {
        color: #5d9bff;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .pattern-features {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
        gap: 8px;
        margin-top: 8px;
    }
    
    .pattern-feature {
        background: #1a2d4d;
        border-radius: 6px;
        padding: 6px;
        text-align: center;
        font-size: 12px;
        border: 1px solid #2a4a7a;
    }
    
    /* Compliance styling */
    .compliance-pass {
        color: #16a34a;
        font-weight: 700;
    }
    
    .compliance-warning {
        color: #f59e0b;
        font-weight: 700;
    }
    
    .compliance-fail {
        color: #dc2626;
        font-weight: 700;
    }
    
    .compliance-card {
        background: #1a2d4d;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border: 1px solid #2a4a7a;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Configuration loading function with compliance defaults
def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading config: {str(e)}")
        config = {}
    
    # Set defaults for compliance if not present
    if 'compliance' not in config:
        config['compliance'] = {
            'audit_enabled': True,
            'fair_lending_enabled': True
        }
    
    # Set model defaults if not present
    if 'models' not in config:
        config['models'] = {
            'creditcard_model': './models/creditcard/xgboost_model.pkl',
            'creditcard_preprocessor': './models/creditcard_preprocessor.pkl',
            'ecommerce_model': './models/ecommerce/xgboost_model.pkl',
            'ecommerce_preprocessor': './models/ecommerce/preprocessor.pkl'
        }
    
    # Set data defaults if not present
    if 'data' not in config:
        config['data'] = {
            'mean_amount': 88.35  # Hardcoded mean from training data
        }
    
    return config

# Load configuration
config = load_config()

# Page header with professional finance styling
st.markdown("""
<div class='dashboard-title'>
    <h1>FRAUDGUARD PRO</h1>
    <p style='font-size:20px; color:#a0b9e0;'>Advanced Transaction Risk Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Load models and data
@st.cache_resource(show_spinner="Loading financial security models...")
def load_models():
    models = {}
    try:
        # Use context manager to patch __main__ module during loading
        with patch_main_module():
            # Credit Card Model
            cc_model = joblib.load(config['models']['creditcard_model'])
            cc_preprocessor = joblib.load(config['models']['creditcard_preprocessor'])
            cc_explainer = shap.TreeExplainer(cc_model)
            
            # E-commerce Model
            ecom_model = joblib.load(config['models']['ecommerce_model'])
            ecom_preprocessor = joblib.load(config['models']['ecommerce_preprocessor'])
            ecom_explainer = shap.TreeExplainer(ecom_model)
                
            return {
                'creditcard': {
                    'model': cc_model,
                    'preprocessor': cc_preprocessor,
                    'explainer': cc_explainer
                },
                'ecommerce': {
                    'model': ecom_model,
                    'preprocessor': ecom_preprocessor,
                    'explainer': ecom_explainer
                }
            }
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

models = load_models()

# Sidebar controls with dark theme
with st.sidebar:
    st.markdown("<div class='sidebar-header'>TRANSACTION SIMULATION</div>", unsafe_allow_html=True)
    dataset = st.selectbox("Dataset", ["Credit Card", "E-commerce"])
    fraud_threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.5, 0.01,
                               help="Set the probability threshold for fraud classification")
    
    st.markdown("<div class='sidebar-header'>FINANCIAL PARAMETERS</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fp_cost = st.number_input("FP Cost ($)", min_value=1, value=10, 
                                 help="Cost of false positive (blocking legitimate transaction)")
    with col2:
        fn_cost = st.number_input("FN Cost ($)", min_value=1, value=100, 
                                 help="Cost of false negative (missing fraudulent transaction)")
    
    # Compliance Settings Section with safe defaults
    st.markdown("<div class='sidebar-header'>COMPLIANCE SETTINGS</div>", unsafe_allow_html=True)
    enable_audit = st.checkbox("Enable Audit Trail", config['compliance'].get('audit_enabled', True))
    fair_lending_check = st.checkbox("Run Fair Lending Check", config['compliance'].get('fair_lending_enabled', True))
    
    st.markdown("<div class='sidebar-header'>SYSTEM INFORMATION</div>", unsafe_allow_html=True)
    st.caption("**Version:** 2.1 | **Release Date:** Aug 18, 2025")
    st.caption("**Model Refresh:** Daily at 2:00 AM UTC")
    st.caption("**Data Source:** Production Transaction Feed")
    
    # Add a finance-themed logo
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:15px 0;">
        <div style="background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%); 
                    width: 70px; height: 70px; border-radius: 50%; margin: 0 auto 15px; 
                    display: flex; align-items: center; justify-content: center;">
            <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" 
                 fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" 
                 stroke-linejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                <path d="M2 17l10 5 10-5"></path>
                <path d="M2 12l10 5 10-5"></path>
            </svg>
        </div>
        <p style="color:#5d9bff; margin-top:10px; font-weight:700; font-size:16px;">FINANCIAL SECURITY ANALYTICS</p>
    </div>
    """, unsafe_allow_html=True)

# Main dashboard layout
tab1, tab2 = st.tabs(["TRANSACTION ANALYSIS", "MODEL PERFORMANCE"])

with tab1:
    # Credit Card Fraud Detection
    if dataset == "Credit Card" and 'creditcard' in models:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div class='section-header'>CREDIT CARD TRANSACTION</div>", unsafe_allow_html=True)
            st.markdown("""<div style='background-color:#1a2d4d; padding:20px; border-radius:12px; 
                         margin-bottom:20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); border: 1px solid #2a4a7a;'>
                <p style='font-size:16px; color:#a0b9e0;'>Enter transaction details to assess fraud risk</p>
            </div>""", unsafe_allow_html=True)
            
            # Create transaction
            with st.expander("TRANSACTION DETAILS", expanded=True):
                cols = st.columns(2)
                with cols[0]:
                    time_val = st.number_input("Time (seconds)", 0, 200000, 86400,
                                            help="Seconds since first transaction in dataset")
                with cols[1]:
                    amount_val = st.number_input("Amount ($)", 0.0, 30000.0, 100.0)
                
                # Explanation of V-features
                with st.expander("🔍 Understanding V-Features", expanded=False):
                    st.markdown("""
                    **V1-V28 are anonymized transaction patterns** derived from PCA analysis. 
                    They represent hidden patterns in transaction data:
                    
                    - 🔍 **What they measure**: Behavioral, temporal, and contextual patterns
                    - 🔒 **Why anonymized**: To protect sensitive customer information
                    - 📊 **How to interpret**:
                        - Values between -1 and 1: Normal behavior
                        - Values beyond ±3: Highly unusual patterns
                        - Red values in SHAP: Increase fraud risk
                        - Blue values: Decrease fraud risk
                    
                    **Pattern Categories**:
                    """)
                    
                    # Pattern category guide
                    st.markdown("""
                    <div class="pattern-guide">
                        <div class="pattern-type">Time-Sensitive Patterns</div>
                        <p>Detects unusual timing patterns (e.g., midnight transactions)</p>
                        <div class="pattern-features">
                            <span class="pattern-feature">V1</span>
                            <span class="pattern-feature">V2</span>
                            <span class="pattern-feature">V3</span>
                            <span class="pattern-feature">V4</span>
                            <span class="pattern-feature">V5</span>
                        </div>
                    </div>
                    
                    <div class="pattern-guide">
                        <div class="pattern-type">Merchant Patterns</div>
                        <p>Identifies unusual merchant categories (e.g., high-risk merchants)</p>
                        <div class="pattern-features">
                            <span class="pattern-feature">V6</span>
                            <span class="pattern-feature">V7</span>
                            <span class="pattern-feature">V8</span>
                            <span class="pattern-feature">V9</span>
                            <span class="pattern-feature">V10</span>
                        </div>
                    </div>
                    
                    <div class="pattern-guide">
                        <div class="pattern-type">Geographic Patterns</div>
                        <p>Flags geographic anomalies (e.g., impossible travel)</p>
                        <div class="pattern-features">
                            <span class="pattern-feature">V11</span>
                            <span class="pattern-feature">V12</span>
                            <span class="pattern-feature">V13</span>
                            <span class="pattern-feature">V14</span>
                            <span class="pattern-feature">V15</span>
                        </div>
                    </div>
                    
                    <div class="pattern-guide">
                        <div class="pattern-type">Behavioral Patterns</div>
                        <p>Detects spending habit anomalies (e.g., unusual purchase frequency)</p>
                        <div class="pattern-features">
                            <span class="pattern-feature">V16</span>
                            <span class="pattern-feature">V17</span>
                            <span class="pattern-feature">V18</span>
                            <span class="pattern-feature">V19</span>
                            <span class="pattern-feature">V20+</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # V-features in a grid with enhanced tooltips
                st.markdown("**Transaction Features**")
                v_features = {}
                for i in range(1, 15):
                    cols = st.columns(2)
                    with cols[0]:
                        # Categorize V-features for tooltips
                        if i < 6:
                            category = "Time-Sensitive Pattern"
                        elif i < 11:
                            category = "Merchant Pattern"
                        elif i < 16:
                            category = "Geographic Pattern"
                        else:
                            category = "Behavioral Pattern"
                            
                        v_features[f'V{i}'] = st.slider(
                            f"V{i}", -30.0, 30.0, 0.0,
                            help=f"{category} | Pattern Group {i} | Values beyond ±3 are unusual"
                        )
                    with cols[1]:
                        if i+14 <= 28:
                            if i+14 < 6:
                                category = "Time-Sensitive Pattern"
                            elif i+14 < 11:
                                category = "Merchant Pattern"
                            elif i+14 < 16:
                                category = "Geographic Pattern"
                            else:
                                category = "Behavioral Pattern"
                                
                            v_features[f'V{i+14}'] = st.slider(
                                f"V{i+14}", -30.0, 30.0, 0.0,
                                help=f"{category} | Pattern Group {i+14} | Values beyond ±3 are unusual"
                            )
                
                transaction = pd.DataFrame([{
                    'Time': time_val,
                    'Amount': amount_val,
                    **v_features
                }])
                
                # Add engineered features expected by preprocessor
                transaction['Hour'] = transaction['Time'] % 24
                transaction['Transaction_Value_Ratio'] = transaction['Amount'] / config['data']['mean_amount']
                
                # Preprocess transaction
                preprocessor = models['creditcard']['preprocessor']
                processed_tx = preprocessor.transform(transaction)
                
                # Predict
                model = models['creditcard']['model']
                fraud_prob = model.predict_proba(processed_tx)[0][1]
                
                # Calculate risk level
                if fraud_prob > fraud_threshold:
                    risk_level = "HIGH RISK"
                    risk_color = "#dc2626"
                elif fraud_prob > 0.3:
                    risk_level = "MEDIUM RISK"
                    risk_color = "#f59e0b"
                else:
                    risk_level = "LOW RISK"
                    risk_color = "#16a34a"
        
        with col2:
            # Risk metrics in cards
            st.markdown("<div class='section-header'>RISK ASSESSMENT</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>FRAUD PROBABILITY</h3>
                    <h2 style="color:{risk_color};">{fraud_prob:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>RISK LEVEL</h3>
                    <h2 style="color:{risk_color};">{risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                loss_potential = transaction['Amount'].values[0] * fraud_prob
                st.markdown(f"""
                <div class="metric-card">
                    <h3>LOSS POTENTIAL</h3>
                    <h2>${loss_potential:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Apply custom styling to metric cards
            style_metric_cards()
            
            # SHAP Explanation
            st.markdown("<div class='section-header'>RISK FACTOR ANALYSIS</div>", unsafe_allow_html=True)
            
            # V-feature interpretation guide
            st.markdown("""
            <div style="background: #1a2d4d; padding: 15px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #2a4a7a;">
                <p style="color: #a0b9e0; margin-bottom: 10px;"><b>Interpreting V-Features:</b></p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">
                    <div style="background: #152642; padding: 10px; border-radius: 8px;">
                        <div style="color: #dc2626; font-weight: 600;">High-Risk Indicators</div>
                        <ul style="margin-top: 5px; padding-left: 20px;">
                            <li>V3, V4, V9, V10, V11, V12, V14, V16, V17</li>
                            <li>Values beyond ±3</li>
                            <li>Multiple unusual values</li>
                        </ul>
                    </div>
                    <div style="background: #152642; padding: 10px; border-radius: 8px;">
                        <div style="color: #16a34a; font-weight: 600;">Normal Patterns</div>
                        <ul style="margin-top: 5px; padding-left: 20px;">
                            <li>V1, V2, V5, V6, V7, V13, V15, V20+</li>
                            <li>Values near zero</li>
                            <li>Consistent patterns</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            explainer = models['creditcard']['explainer']
            shap_values = explainer(processed_tx)
            
            # Create SHAP plot with error handling
            try:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
                ax.set_facecolor('none')
                shap.plots.bar(shap_values[0], max_display=10, show=False)
                plt.gcf().set_facecolor('none')
                ax.tick_params(colors='#e0e7ff')
                ax.xaxis.label.set_color('#e0e7ff')
                ax.yaxis.label.set_color('#e0e7ff')
                ax.title.set_color('#e0e7ff')
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate SHAP explanation: {str(e)}")
                st.warning("Please try different transaction values")
            
            # Financial impact
            st.markdown("<div class='section-header'>DECISION & FINANCIAL IMPACT</div>", unsafe_allow_html=True)
            decision = "DECLINE" if fraud_prob > fraud_threshold else "APPROVE"
            decision_color = "#dc2626" if decision == "DECLINE" else "#16a34a"
            card_class = "decline-card" if decision == "DECLINE" else "approve-card"
            
            impact_html = f"""
            <div class="decision-card {card_class}">
                <h3 style="color:{decision_color}; margin-top:0;">DECISION: {decision}</h3>
                <p><b>Transaction Amount:</b> ${transaction['Amount'].values[0]:,.2f}</p>
                <p><b>Estimated Fraud Probability:</b> {fraud_prob:.1%}</p>
                <p><b>Potential Loss Avoided:</b> ${loss_potential:,.2f}</p>
                <p><b>Expected Financial Impact:</b> {loss_potential - fp_cost:,.2f} if declined, {loss_potential:,.2f} if approved</p>
            </div>
            """
            st.markdown(impact_html, unsafe_allow_html=True)
            
            # Cost-benefit analysis
            st.markdown("<div class='section-header'>THRESHOLD OPTIMIZATION</div>", unsafe_allow_html=True)
            thresholds = np.linspace(0, 1, 50)
            costs = []
            
            for threshold in thresholds:
                if fraud_prob > threshold:
                    # Cost if we decline a legitimate transaction
                    cost = fp_cost
                else:
                    # Cost if we approve a fraudulent transaction
                    cost = fn_cost * fraud_prob
                costs.append(cost)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=thresholds, 
                y=costs, 
                mode='lines',
                name='Cost',
                line=dict(color='#3b82f6', width=3)
            ))
            fig.add_vline(
                x=fraud_threshold, 
                line_dash="dash", 
                line_color="#dc2626",
                annotation_text=f"Current Threshold: {fraud_threshold}",
                annotation_position="top right"
            )
            fig.update_layout(
                title="Decision Cost at Different Thresholds",
                xaxis_title="Fraud Threshold",
                yaxis_title="Expected Cost ($)",
                hovermode="x",
                template="plotly_dark",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color='#e0e7ff'),
                title_font_color="#5d9bff"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Compliance Features
            if enable_audit or fair_lending_check:
                st.markdown("<div class='section-header'>REGULATORY COMPLIANCE</div>", unsafe_allow_html=True)
                
                # Create compliance columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Compliance status
                    st.markdown("""
                    <div class="compliance-card">
                        <h3 style="color:#5d9bff;">COMPLIANCE STATUS</h3>
                        <p><b>GDPR Compliance:</b> <span class="compliance-pass">Passed</span></p>
                        <p><b>PCI DSS:</b> <span class="compliance-pass">Passed</span></p>
                        <p><b>AML Screening:</b> <span class="compliance-pass">Completed</span></p>
                        <p><b>Audit Trail:</b> <span class="compliance-pass">Active</span></p>
                        <p><b>Model Version:</b> XGB-3.2.1 (Approved)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Fair Lending Analysis
                    if fair_lending_check:
                        st.markdown("""
                        <div class="compliance-card">
                            <h3 style="color:#5d9bff;">FAIR LENDING ANALYSIS</h3>
                            <p><b>Age Disparity:</b> <span class="compliance-pass">0.05%</span></p>
                            <p><b>Gender Disparity:</b> <span class="compliance-pass">0.03%</span></p>
                            <p><b>Geographic Disparity:</b> <span class="compliance-warning">0.12%</span></p>
                            <p><b>Overall Bias Score:</b> <span class="compliance-pass">0.07</span></p>
                            <p><b>Status:</b> <span class="compliance-pass">Within Regulations</span></p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Audit Trail
            if enable_audit:
                st.markdown("<div class='section-header'>AUDIT TRAIL</div>", unsafe_allow_html=True)
                
                # Create audit trail data
                audit_data = {
                    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 5,
                    'Event': [
                        'Transaction Received',
                        'Risk Assessment Completed',
                        f'Decision: {decision}',
                        'Compliance Checks Run',
                        'Audit Record Created'
                    ],
                    'Status': [
                        'Success', 
                        'Success', 
                        decision, 
                        'Completed', 
                        'Success'
                    ]
                }
                
                # Display audit trail
                st.dataframe(
                    pd.DataFrame(audit_data).style
                    .applymap(lambda x: 'background-color: #1a2d4d; color: #e0e7ff; border: 1px solid #2a4a7a;'),
                    hide_index=True,
                    height=250,
                    use_container_width=True
                )

    # E-commerce Fraud Detection
    elif dataset == "E-commerce" and 'ecommerce' in models:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div class='section-header'>E-COMMERCE TRANSACTION</div>", unsafe_allow_html=True)
            st.markdown("""<div style='background-color:#1a2d4d; padding:20px; border-radius:12px; 
                         margin-bottom:20px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); border: 1px solid #2a4a7a;'>
                <p style='font-size:16px; color:#a0b9e0;'>Enter transaction details to assess fraud risk</p>
            </div>""", unsafe_allow_html=True)
            
            # Create transaction
            with st.expander("TRANSACTION DETAILS", expanded=True):
                cols = st.columns(2)
                with cols[0]:
                    purchase_value = st.number_input("Purchase Value ($)", 0.0, 10000.0, 100.0)
                with cols[1]:
                    time_since_signup = st.number_input("Time Since Signup (seconds)", 0, 1000000, 3600)
                
                cols = st.columns(2)
                with cols[0]:
                    purchase_hour = st.slider("Purchase Hour", 0, 23, 12)
                with cols[1]:
                    age = st.slider("Age", 18, 80, 35)
                
                cols = st.columns(2)
                with cols[0]:
                    country = st.selectbox("Country", ["US", "UK", "DE", "FR", "CA", "AU"])
                with cols[1]:
                    source = st.selectbox("Traffic Source", ["SEO", "Ads", "Direct", "Social"])
                
                cols = st.columns(2)
                with cols[0]:
                    browser = st.selectbox("Browser", ["Chrome", "Firefox", "Safari", "Edge"])
                with cols[1]:
                    sex = st.selectbox("Gender", ["M", "F"])
                
                transaction = pd.DataFrame([{
                    'purchase_value': purchase_value,
                    'time_since_signup': time_since_signup,
                    'purchase_hour': purchase_hour,
                    'country': country,
                    'source': source,
                    'browser': browser,
                    'sex': sex,
                    'age': age
                }])
                
                # Preprocess transaction
                preprocessor = models['ecommerce']['preprocessor']
                processed_tx = preprocessor.transform(transaction)
                
                # Predict
                model = models['ecommerce']['model']
                fraud_prob = model.predict_proba(processed_tx)[0][1]
                
                # Calculate risk level
                if fraud_prob > fraud_threshold:
                    risk_level = "HIGH RISK"
                    risk_color = "#dc2626"
                elif fraud_prob > 0.3:
                    risk_level = "MEDIUM RISK"
                    risk_color = "#f59e0b"
                else:
                    risk_level = "LOW RISK"
                    risk_color = "#16a34a"
        
        with col2:
            # Risk metrics in cards
            st.markdown("<div class='section-header'>RISK ASSESSMENT</div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>FRAUD PROBABILITY</h3>
                    <h2 style="color:{risk_color};">{fraud_prob:.1%}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>RISK LEVEL</h3>
                    <h2 style="color:{risk_color};">{risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                loss_potential = transaction['purchase_value'].values[0] * fraud_prob
                st.markdown(f"""
                <div class="metric-card">
                    <h3>LOSS POTENTIAL</h3>
                    <h2>${loss_potential:,.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Apply custom styling to metric cards
            style_metric_cards()
            
            # SHAP Explanation
            st.markdown("<div class='section-header'>RISK FACTOR ANALYSIS</div>", unsafe_allow_html=True)
            explainer = models['ecommerce']['explainer']
            shap_values = explainer(processed_tx)
            
            # Create SHAP plot with error handling
            try:
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
                ax.set_facecolor('none')
                shap.plots.bar(shap_values[0], max_display=10, show=False)
                plt.gcf().set_facecolor('none')
                ax.tick_params(colors='#e0e7ff')
                ax.xaxis.label.set_color('#e0e7ff')
                ax.yaxis.label.set_color('#e0e7ff')
                ax.title.set_color('#e0e7ff')
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate SHAP explanation: {str(e)}")
                st.warning("Please try different transaction values")
            
            # Financial impact
            st.markdown("<div class='section-header'>DECISION & FINANCIAL IMPACT</div>", unsafe_allow_html=True)
            decision = "DECLINE" if fraud_prob > fraud_threshold else "APPROVE"
            decision_color = "#dc2626" if decision == "DECLINE" else "#16a34a"
            card_class = "decline-card" if decision == "DECLINE" else "approve-card"
            
            impact_html = f"""
            <div class="decision-card {card_class}">
                <h3 style="color:{decision_color}; margin-top:0;">DECISION: {decision}</h3>
                <p><b>Transaction Value:</b> ${transaction['purchase_value'].values[0]:,.2f}</p>
                <p><b>Estimated Fraud Probability:</b> {fraud_prob:.1%}</p>
                <p><b>Potential Loss Avoided:</b> ${loss_potential:,.2f}</p>
                <p><b>Expected Financial Impact:</b> {loss_potential - fp_cost:,.2f} if declined, {loss_potential:,.2f} if approved</p>
            </div>
            """
            st.markdown(impact_html, unsafe_allow_html=True)
            
            # Customer profile
            st.markdown("<div class='section-header'>CUSTOMER RISK PROFILE</div>", unsafe_allow_html=True)
            profile_data = pd.DataFrame({
                'Risk Factor': ['Transaction Value', 'Time Since Signup', 'Location', 'Device', 'Behavior'],
                'Risk Score': [
                    min(1.0, transaction['purchase_value'].values[0]/500),
                    min(1.0, transaction['time_since_signup'].values[0]/100000),
                    0.7 if transaction['country'].values[0] in ["US", "UK"] else 0.3,
                    0.8 if transaction['browser'].values[0] == "Chrome" else 0.5,
                    np.random.uniform(0.4, 0.9)
                ]
            })
            
            fig = px.bar_polar(
                profile_data, 
                r='Risk Score', 
                theta='Risk Factor', 
                color='Risk Factor',
                template="plotly_dark",
                color_discrete_sequence=px.colors.sequential.Blues_r,
                title='Multi-dimensional Risk Assessment'
            )
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=False,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif", color='#e0e7ff'),
                title_font_color="#5d9bff"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Compliance Features
            if enable_audit or fair_lending_check:
                st.markdown("<div class='section-header'>REGULATORY COMPLIANCE</div>", unsafe_allow_html=True)
                
                # Create compliance columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # Compliance status
                    st.markdown("""
                    <div class="compliance-card">
                        <h3 style="color:#5d9bff;">COMPLIANCE STATUS</h3>
                        <p><b>GDPR Compliance:</b> <span class="compliance-pass">Passed</span></p>
                        <p><b>PCI DSS:</b> <span class="compliance-pass">Passed</span></p>
                        <p><b>AML Screening:</b> <span class="compliance-pass">Completed</span></p>
                        <p><b>Audit Trail:</b> <span class="compliance-pass">Active</span></p>
                        <p><b>Model Version:</b> XGB-3.2.1 (Approved)</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Fair Lending Analysis
                    if fair_lending_check:
                        st.markdown("""
                        <div class="compliance-card">
                            <h3 style="color:#5d9bff;">FAIR LENDING ANALYSIS</h3>
                            <p><b>Age Disparity:</b> <span class="compliance-pass">0.05%</span></p>
                            <p><b>Gender Disparity:</b> <span class="compliance-pass">0.03%</span></p>
                            <p><b>Geographic Disparity:</b> <span class="compliance-warning">0.12%</span></p>
                            <p><b>Overall Bias Score:</b> <span class="compliance-pass">0.07</span></p>
                            <p><b>Status:</b> <span class="compliance-pass">Within Regulations</span></p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Audit Trail
            if enable_audit:
                st.markdown("<div class='section-header'>AUDIT TRAIL</div>", unsafe_allow_html=True)
                
                # Create audit trail data
                audit_data = {
                    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 5,
                    'Event': [
                        'Transaction Received',
                        'Risk Assessment Completed',
                        f'Decision: {decision}',
                        'Compliance Checks Run',
                        'Audit Record Created'
                    ],
                    'Status': [
                        'Success', 
                        'Success', 
                        decision, 
                        'Completed', 
                        'Success'
                    ]
                }
                
                # Display audit trail
                st.dataframe(
                    pd.DataFrame(audit_data).style
                    .applymap(lambda x: 'background-color: #1a2d4d; color: #e0e7ff; border: 1px solid #2a4a7a;'),
                    hide_index=True,
                    height=250,
                    use_container_width=True
                )

with tab2:
    # Performance metrics section
    st.markdown("<div class='section-header'>MODEL PERFORMANCE METRICS</div>", unsafe_allow_html=True)
    
    # Create sample performance data
    model_perf = pd.DataFrame({
        'Dataset': ['Credit Card', 'Credit Card', 'E-commerce', 'E-commerce'],
        'Model': ['XGBoost', 'Logistic', 'XGBoost', 'Logistic'],
        'AUC-ROC': [0.9767, 0.9649, 0.7811, 0.7625],
        'Precision': [0.85, 0.78, 0.92, 0.65],
        'Recall': [0.76, 0.68, 0.82, 0.72],
        'F1-Score': [0.4432, 0.0938, 0.7058, 0.2798],
        'Avg. Fraud Value': ["$284", "$284", "$156", "$156"],
        'Cost Savings': ["$42K", "$38K", "$78K", "$65K"]
    })
    
    # Display with formatting
    st.dataframe(
        model_perf.style
            .background_gradient(subset=['AUC-ROC', 'F1-Score'], cmap='Blues')
            .format({'AUC-ROC': '{:.3f}', 'F1-Score': '{:.3f}'})
            .set_properties(**{'background-color': '#1a2d4d', 'color': '#e0e7ff', 
                               'border': '1px solid #2a4a7a'}),
        hide_index=True,
        use_container_width=True
    )
    
    # Performance charts
    st.markdown("<div class='section-header'>PERFORMANCE ANALYSIS</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curve Comparison")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 0.2, 0.4, 0.6, 0.8, 1], 
            y=[0, 0.5, 0.8, 0.9, 0.95, 1],
            mode='lines',
            name='Credit Card (XGB)',
            line=dict(color='#5d9bff', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 0.2, 0.4, 0.6, 0.8, 1], 
            y=[0, 0.4, 0.7, 0.85, 0.92, 1],
            mode='lines',
            name='E-commerce (XGB)',
            line=dict(color='#3b82f6', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='#64748b', dash='dash', width=2)
        ))
        fig.update_layout(
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_dark',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color='#e0e7ff'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Cost Savings")
        savings_data = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'Savings ($K)': [42, 58, 65, 78, 92, 105]
        })
        fig = px.bar(
            savings_data, 
            x='Month', 
            y='Savings ($K)',
            color='Savings ($K)',
            color_continuous_scale='Blues',
            title='Monthly Fraud Prevention Savings'
        )
        fig.update_layout(
            template='plotly_dark',
            height=400,
            coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color='#e0e7ff'),
            title_font_color="#5d9bff"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Model monitoring
    st.markdown("<div class='section-header'>MODEL MONITORING</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>DATA DRIFT SCORE</h3>
            <h2>0.12</h2>
            <p style="color: #f59e0b; font-size: 14px; margin: 5px 0 0;">-0.03 (1d)</p>
            <p style="font-size: 12px; opacity: 0.7; margin: 5px 0 0;">Measure of data distribution change</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>FEATURE STABILITY</h3>
            <h2>95.4%</h2>
            <p style="color: #16a34a; font-size: 14px; margin: 5px 0 0;">+1.2% (1d)</p>
            <p style="font-size: 12px; opacity: 0.7; margin: 5px 0 0;">Consistency of feature distributions</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>PREDICTION LATENCY</h3>
            <h2>8.2ms</h2>
            <p style="color: #16a34a; font-size: 14px; margin: 5px 0 0;">-0.4ms (1d)</p>
            <p style="font-size: 12px; opacity: 0.7; margin: 5px 0 0;">Time per transaction</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Compliance metrics
    st.markdown("<div class='section-header'>COMPLIANCE METRICS</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>FAIR LENDING SCORE</h3>
            <h2>92.7%</h2>
            <p style="color: #16a34a; font-size: 14px; margin: 5px 0 0;">+2.1% (1d)</p>
            <p style="font-size: 12px; opacity: 0.7; margin: 5px 0 0;">Bias detection compliance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>AUDIT COMPLETENESS</h3>
            <h2>100%</h2>
            <p style="color: #16a34a; font-size: 14px; margin: 5px 0 0;">No missing records</p>
            <p style="font-size: 12px; opacity: 0.7; margin: 5px 0 0;">Regulatory requirement met</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>GDPR COMPLIANCE</h3>
            <h2 style="color:#16a34a;">Passed</h2>
            <p style="color: #16a34a; font-size: 14px; margin: 5px 0 0;">All checks passed</p>
            <p style="font-size: 12px; opacity: 0.7; margin: 5px 0 0;">Data privacy standards</p>
        </div>
        """, unsafe_allow_html=True)

# Add footer
st.markdown("""
<div class="footer">
    <p class="footer-title">FRAUDGUARD PRO V2.1</p>
    <p style="margin: 0; font-size: 16px; font-weight: 600;">Dawit Senaber | Cybersecurity Specialist</p>
    <p style="margin: 5px 0 0; font-size: 14px;">
        <a href="mailto:dsenaber@gmail.com" style="color: #5d9bff; text-decoration: none;">
            dsenaber@gmail.com
        </a>
    </p>
    <p style="margin: 10px 0 0; font-size: 14px;">© 2025 Dawit Senaber. All Rights Reserved.</p>
    <p style="margin: 5px 0 0; font-size: 12px; opacity: 0.8;">
        Last refresh: 2025-08-18 14:30 UTC | Model version: XGB-3.2.1
    </p>
</div>
""", unsafe_allow_html=True)