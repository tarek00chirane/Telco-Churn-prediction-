import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# ─── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
    
    
)


# ─── Custom CSS for better appearance ─────────────────────────────────────────
st.markdown("""
    <style>
        .main { max-width: 1200px; margin: 0 auto; }
        .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
        .risk-high { color: #c0392b; font-weight: bold; }
        .risk-low { color: #1e8449; font-weight: bold; }
        .risk-medium { color: #d68910; font-weight: bold; }
        h1 { color: #1f4e79; }
        h2 { color: #2e75b6; }
    </style>
""", unsafe_allow_html=True)

# ─── Load pipeline ────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    try:
       
        # Get the directory where this script is located
        current_dir = Path(__file__).parent
        
        # Try multiple possible locations (in order of likelihood)
        possible_paths = [
            current_dir / 'churn_production_pipeline.pkl',           # Same folder as app.py
            current_dir / '..' / 'churn_production_pipeline.pkl',    # One level up (repo root)
            Path('churn_production_pipeline.pkl'),                   # Working directory root
            Path('/app/churn_production_pipeline.pkl'),              # Streamlit Cloud absolute path
            Path('/mount/src/your-repo-name/churn_production_pipeline.pkl'),  # GitHub Codespaces
        ]
        
        # Try each path until we find the file
        pipeline = None
        for path in possible_paths:
            if path.exists():
               # st.info(f"✅ Found pipeline at:  {path}")
                pipeline = joblib.load(path)
                break
        
        if pipeline is None:
            # Debug: Show what files exist in the current directory
            st.error("❌ Pipeline file 'churn_production_pipeline.pkl' not found!")
            st.write("📁 Files in current directory:", os.listdir('.'))
            st.write("📁 Files in parent directory:", os.listdir('..') if os.path.exists('..') else "N/A")
            st.write("📁 Current working directory:", os.getcwd())
            return None
        
        return pipeline
        
    except Exception as e:
        st.error(f"❌ Error loading pipeline: {str(e)}")
        return None

pipeline = load_pipeline()

# ─── Title and intro ──────────────────────────────────────────────────────────
st.title(" Telco Customer Churn Predictor")
st.markdown("**Predict customer churn risk and get actionable recommendations**")
st.markdown("<p style='text-align: right; color: gray; font-size: 11px;'><i><b>Prepared by Tarek Chirane</b></i></p>", unsafe_allow_html=True)
st.divider()

# ─── Sidebar: Input section ───────────────────────────────────────────────────
st.sidebar.header("Customer Profile")

# Basic info
tenure = st.sidebar.slider(
    "Tenure (months)",
    min_value=0, max_value=72, value=24,
    help="How long has the customer been with us?"
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=18.0, max_value=119.0, value=65.0, step=0.5,
    help="What is the customer's current monthly bill?"
)

total_charges = st.sidebar.slider(
    "Total Charges ($)",
    min_value=18.0, max_value=8684.0, value=1500.0, step=10.0,
    help="Total amount the customer has paid to date"
)

st.sidebar.divider()
st.sidebar.header("Services")

# Services (count how many are active)
phone_service = st.sidebar.checkbox("Phone Service", value=True)
multiple_lines = st.sidebar.checkbox("Multiple Lines", value=False)
internet_service = st.sidebar.selectbox(
    "Internet Service Type",
    ["No", "DSL", "Fiber Optic"],
    help="None, DSL, or Fiber Optic"
)
online_security = st.sidebar.checkbox("Online Security", value=False)
online_backup = st.sidebar.checkbox("Online Backup", value=False)
device_protection = st.sidebar.checkbox("Device Protection", value=False)
tech_support = st.sidebar.checkbox("Tech Support", value=False)
streaming_tv = st.sidebar.checkbox("Streaming TV", value=False)
streaming_movies = st.sidebar.checkbox("Streaming Movies", value=False)

st.sidebar.divider()
st.sidebar.header("Contract & Billing")

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
    help="Most important predictor of churn!"
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    help="Automatic payments indicate stronger commitment"
)

paperless_billing = st.sidebar.checkbox("Paperless Billing", value=False)

# ─── Feature Engineering ──────────────────────────────────────────────────────
# This is the EXACT logic from your notebook

# Count active services
num_active_services = sum([
    phone_service,
    multiple_lines,
    internet_service != "No",
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies
])

# Avoid division by zero
safe_tenure = tenure if tenure > 0 else 0.5
safe_total = total_charges if total_charges > 0 else 0.5

# Calculate engineered features
avg_monthly_charges = total_charges / (tenure + 1)
charge_trend = monthly_charges - avg_monthly_charges
contract_risk = monthly_charges / (tenure + 1)
monthly_to_total_ratio = monthly_charges / (total_charges + 1)
avg_service_usage = num_active_services / 9.0

# Log transform the skewed features
contract_risk_log = np.log1p(contract_risk)
monthly_to_total_ratio_log = np.log1p(monthly_to_total_ratio)

# ─── Encode categorical variables ─────────────────────────────────────────────
# These mappings match your training data

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract_encoded = contract_map[contract]

internet_map = {"No": 0, "DSL": 1, "Fiber Optic": 2}
internet_encoded = internet_map[internet_service]

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}
payment_encoded = payment_map[payment_method]

# Binary features
online_security_binary = 1 if online_security else 0
tech_support_binary = 1 if tech_support else 0
paperless_binary = 1 if paperless_billing else 0

# ─── Create feature array in correct order ────────────────────────────────────
# This MUST match the order used when training the pipeline
feature_array = np.array([
    tenure,
    monthly_charges,
    contract_encoded,
    internet_encoded,
    online_security_binary,
    tech_support_binary,
    paperless_binary,
    payment_encoded,
    contract_risk_log,
    monthly_to_total_ratio_log,
    num_active_services
]).reshape(1, -1)

# ─── Make prediction ──────────────────────────────────────────────────────────
if pipeline is not None:
    churn_probability = pipeline.predict_proba(feature_array)[0][1]
    churn_prediction = pipeline.predict(feature_array)[0]
else:
    churn_probability = 0.5
    churn_prediction = 0

# ─── Risk classification ──────────────────────────────────────────────────────
if churn_probability < 0.30:
    risk_level = "🟢 LOW RISK"
    risk_class = "risk-low"
    risk_color = "#1e8449"
elif churn_probability < 0.60:
    risk_level = "🟡 MEDIUM RISK"
    risk_class = "risk-medium"
    risk_color = "#d68910"
else:
    risk_level = "🔴 HIGH RISK"
    risk_class = "risk-high"
    risk_color = "#c0392b"

# ─── Main content: Results ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Churn Probability",
        value=f"{churn_probability*100:.1f}%",
        delta=None
    )

with col2:
    st.markdown(f"<div style='text-align: center; padding: 20px;'><h3 style='color: {risk_color};'>{risk_level}</h3></div>", unsafe_allow_html=True)

with col3:
    st.metric(
        label="Prediction",
        value="CHURN" if churn_prediction == 1 else "STAY",
        delta=None
    )

st.divider()

# ─── Business recommendations ─────────────────────────────────────────────────
st.header("Recommended Actions")

recommendations = []

# Rule 1: Contract type
if contract == "Month-to-month":
    recommendations.append(
        ("🔴 CRITICAL", "Contract Migration", 
         "Customer on month-to-month contract. Offer incentive to migrate to 1-year or 2-year contract.")
    )
elif contract == "One year":
    recommendations.append(
        ("🟡 MEDIUM", "Upgrade Path", 
         "Offer upgrade to 2-year contract before renewal to increase lock-in.")
    )

# Rule 2: Tenure
if tenure < 6:
    recommendations.append(
        ("🔴 CRITICAL", "Early Engagement", 
         "Customer is very new (<6 months). Proactive onboarding call within 48 hours.")
    )
elif tenure < 12:
    recommendations.append(
        ("🟡 MEDIUM", "First-Year Check-in", 
         "Customer approaching 12-month milestone. Confirm satisfaction before churn window.")
    )

# Rule 3: Monthly charges vs tenure
if monthly_charges > 80 and tenure < 12:
    recommendations.append(
        ("🔴 CRITICAL", "High Bill Alert", 
         "New customer paying premium price. Risk of billing shock. Offer bundle discount.")
    )

# Rule 4: Services
if num_active_services < 3:
    recommendations.append(
        ("🟡 MEDIUM", "Service Adoption", 
         f"Customer using only {num_active_services} services. Recommend bundling (security, backup, support).")
    )
elif num_active_services >= 7:
    recommendations.append(
        ("🟢 LOW", "Loyalty Program", 
         "Customer highly engaged with {num_active_services} services. Good retention candidate.")
    )

# Rule 5: Security & Support
if not online_security and not tech_support and internet_service == "Fiber Optic":
    recommendations.append(
        ("🔴 CRITICAL", "Add Protective Services", 
         "Fiber Optic customer without security/support. Offer free 3-month trial.")
    )

if not recommendations:
    recommendations.append(
        ("🟢 LOW", "Standard Monitoring", 
         "Customer profile is healthy. Continue standard engagement.")
    )

# Display recommendations
for priority, title, description in recommendations:
    st.info(f"**{priority} — {title}**\n{description}")

st.divider()

# ─── Feature importance table ─────────────────────────────────────────────────
st.header("Customer Profile Summary")

profile_data = {
    "Attribute": [
        "Tenure",
        "Monthly Charges",
        "Total Charges",
        "Active Services",
        "Contract Type",
        "Internet Type",
        "Online Security",
        "Tech Support",
        "Payment Method",
        "Paperless Billing"
    ],
    "Value": [
        f"{tenure} months",
        f"${monthly_charges:.2f}",
        f"${total_charges:.2f}",
        f"{num_active_services}/9",
        contract,
        internet_service,
        "✅ Yes" if online_security else "❌ No",
        "✅ Yes" if tech_support else "❌ No",
        payment_method,
        "✅ Yes" if paperless_billing else "❌ No"
    ]
}

st.dataframe(profile_data, use_container_width=True, hide_index=True)

st.divider()

# ─── Engineered features (for transparency) ───────────────────────────────────
with st.expander(" Engineered Features (Advanced)"):
    eng_data = {
        "Feature": [
            "Avg Monthly Charges",
            "Charge Trend",
            "Contract Risk (log)",
            "Monthly-to-Total Ratio (log)",
            "Service Usage Rate"
        ],
        "Value": [
            f"${avg_monthly_charges:.2f}",
            f"${charge_trend:+.2f}",
            f"{contract_risk_log:.4f}",
            f"{monthly_to_total_ratio_log:.4f}",
            f"{avg_service_usage*100:.1f}%"
        ],
        "Interpretation": [
            "Historical average spend per month",
            "Current bill vs historical (+ means increase)",
            "High risk when new + expensive (most important!)",
            "New customer indicator with high bill",
            "Fraction of available services used"
        ]
    }
    st.dataframe(eng_data, use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style='text-align: center; color: gray; font-size: 12px;'>
    Telco Churn Prediction Model | Built with Streamlit | Model AUC-ROC: 0.847
</p>
""", unsafe_allow_html=True)
