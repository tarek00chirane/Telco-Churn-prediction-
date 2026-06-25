import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="wide",
)

# ─── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main { max-width: 1200px; margin: 0 auto; }
        .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 10px; }
        h1 { color: #1f4e79; }
        h2 { color: #2e75b6; }
    </style>
""", unsafe_allow_html=True)

# ─── Load Pipeline ──────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    """Load the trained churn prediction pipeline from disk."""
    current_dir = Path(__file__).parent
    possible_paths = [
        current_dir / 'churn_production_pipeline.pkl',
        current_dir / '..' / 'churn_production_pipeline.pkl',
        Path('churn_production_pipeline.pkl'),
        Path('/app/churn_production_pipeline.pkl'),
        Path('/mount/src/your-repo-name/churn_production_pipeline.pkl'),
    ]
    for path in possible_paths:
        if path.exists():
            return joblib.load(path), path
    return None, None

pipeline, pipeline_path = load_pipeline()

# ─── Page Title ─────────────────────────────────────────────────────────────
st.title("Telco Customer Churn Predictor")
st.markdown("**ML-powered churn risk assessment. Predict customer churn risk and get actionable recommendations**")
st.divider()

# ─── Error Handling ─────────────────────────────────────────────────────────
if pipeline is None:
    st.error("Pipeline file not found. Place churn_production_pipeline.pkl in the app directory.")
    st.stop()



# ─── Categorical Encodings ──────────────────────────────────────────────────
# These mappings must match the trained pipeline's LabelEncoder values
CATEGORICAL_ENCODINGS = {
    'Contract': {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2
    },
    'InternetService': {
        "DSL": 0,
        "Fiber optic": 1,
        "No": 2
    },
    'OnlineSecurity': {
        "No": 0,
        "Yes": 1
    },
    'PaymentMethod': {
        "Bank transfer (automatic)": 0,
        "Credit card (automatic)": 1,
        "Electronic check": 2,
        "Mailed check": 3
    },
    'PaperlessBilling': {
        "No": 0,
        "Yes": 1
    },
    'TechSupport': {
        "No": 0,
        "Yes": 1
    }
}

# ─── Sidebar: Customer Inputs ───────────────────────────────────────────────
st.sidebar.header("Customer Profile")

tenure = st.sidebar.slider(
    "Tenure (months)",
    min_value=0, max_value=72, value=24,
    help="How long has the customer been with us?"
)

monthly_charges = st.sidebar.slider(
    "Monthly Charges ($)",
    min_value=18.0, max_value=119.0, value=65.0, step=0.5,
    help="Current monthly bill amount"
)

total_charges = st.sidebar.slider(
    "Total Charges ($)",
    min_value=18.0, max_value=8684.0, value=1560.0, step=10.0,
    help="Total amount paid to date"
)

st.sidebar.divider()
st.sidebar.header("Services")

internet_service = st.sidebar.selectbox(
    "Internet Service Type",
    ["DSL", "Fiber optic", "No"],
)
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])

st.sidebar.divider()
st.sidebar.header("Contract & Billing")

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"],
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
)

paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

# ─── Feature Engineering ────────────────────────────────────────────────────
# Compute derived features from raw inputs
tenure_val = tenure
monthly_charges_val = monthly_charges

avg_monthly_charges = total_charges / (tenure + 1)
charge_trend = monthly_charges - avg_monthly_charges
contract_risk = monthly_charges / (tenure + 1)
monthly_to_total_ratio = monthly_charges / (total_charges + 1)

# Apply log transformation for skewed features
contract_risk_log = np.log1p(contract_risk)
monthly_to_total_ratio_log = np.log1p(monthly_to_total_ratio)

# ─── Encode Categorical Variables ───────────────────────────────────────────
contract_encoded = CATEGORICAL_ENCODINGS['Contract'][contract]
internet_encoded = CATEGORICAL_ENCODINGS['InternetService'][internet_service]
online_security_encoded = CATEGORICAL_ENCODINGS['OnlineSecurity'][online_security]
payment_encoded = CATEGORICAL_ENCODINGS['PaymentMethod'][payment_method]
paperless_encoded = CATEGORICAL_ENCODINGS['PaperlessBilling'][paperless_billing]
tech_support_encoded = CATEGORICAL_ENCODINGS['TechSupport'][tech_support]

# ─── Build Feature Array ───────────────────────────────────────────────────
# Features MUST be in exact order expected by the trained pipeline
feature_array = np.array([
    tenure_val,
    monthly_charges_val,
    contract_encoded,
    contract_risk_log,
    charge_trend,
    monthly_to_total_ratio_log,
    internet_encoded,
    online_security_encoded,
    payment_encoded,
    paperless_encoded,
    tech_support_encoded,
]).reshape(1, -1)

# ─── Make Prediction ────────────────────────────────────────────────────────
try:
    churn_probability = pipeline.predict_proba(feature_array)[0][1]
    churn_prediction = pipeline.predict(feature_array)[0]
    
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ─── Determine Risk Level ───────────────────────────────────────────────────
if churn_probability < 0.30:
    risk_level = "LOW RISK"
    risk_color = "#1e8449"
    risk_indicator = "🟢"
elif churn_probability < 0.60:
    risk_level = "MEDIUM RISK"
    risk_color = "#d68910"
    risk_indicator = "🟡"
else:
    risk_level = "HIGH RISK"
    risk_color = "#c0392b"
    risk_indicator = "🔴"

# ─── Display Prediction Results ─────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("Churn Probability", f"{churn_probability*100:.1f}%")
with col2:
    st.markdown(f"<h2 style='text-align: center; color:{risk_color}; margin-top: 20px;'>{risk_indicator}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color:{risk_color}; font-weight: bold; margin-top: -10px;'>{risk_level}</p>", unsafe_allow_html=True)
with col3:
    st.metric("Prediction", "CHURN" if churn_prediction == 1 else "STAY")

st.divider()

# ─── Recommendations ────────────────────────────────────────────────────────
st.header("Recommended Actions")
recommendations = []

# Rule 1: Contract type analysis
if contract == "Month-to-month":
    recommendations.append({
        "severity": "CRITICAL",
        "title": "Contract Migration",
        "description": "Offer incentive to migrate to 1-year or 2-year contract."
    })
elif contract == "One year":
    recommendations.append({
        "severity": "MEDIUM",
        "title": "Upgrade Path",
        "description": "Offer upgrade to 2-year contract before renewal."
    })

# Rule 2: Tenure-based engagement
if tenure < 6:
    recommendations.append({
        "severity": "CRITICAL",
        "title": "Early Engagement",
        "description": "Customer is very new – proactive onboarding call within 48 hours."
    })
elif tenure < 12:
    recommendations.append({
        "severity": "MEDIUM",
        "title": "First-Year Check-in",
        "description": "Confirm satisfaction before churn window."
    })

# Rule 3: New customer with high bill
if monthly_charges > 80 and tenure < 12:
    recommendations.append({
        "severity": "CRITICAL",
        "title": "High Bill Alert",
        "description": "New customer paying premium price – offer bundle discount."
    })

# Rule 4: Missing protective services
if online_security == "No" and tech_support == "No":
    recommendations.append({
        "severity": "MEDIUM",
        "title": "Add Protective Services",
        "description": "Customer lacks security and support – offer free trial."
    })

# Rule 5: Highest-risk archetype
if monthly_charges > 70 and contract == "Month-to-month" and tenure < 6:
    recommendations.append({
        "severity": "CRITICAL",
        "title": "Price Sensitivity Alert",
        "description": "High-risk archetype – immediate retention call needed."
    })

# Default recommendation
if not recommendations:
    recommendations.append({
        "severity": "LOW",
        "title": "Standard Monitoring",
        "description": "Customer profile is healthy — no critical churn signals detected."
    })

# Display recommendations with professional styling
for rec in recommendations:
    if "CRITICAL" in rec["severity"]:
        border_color = "#c0392b"
        bg_color = "#fde8e8"
    elif "MEDIUM" in rec["severity"]:
        border_color = "#f39c12"
        bg_color = "#fef9e7"
    else:
        border_color = "#27ae60"
        bg_color = "#eafaf1"
    
    st.markdown(f"""
        <div style="
            background-color: {bg_color};
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 12px;
            border-left: 6px solid {border_color};
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        ">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                <span style="font-size: 16px; font-weight: bold;">{rec['severity']}</span>
                <span style="font-size: 15px; font-weight: bold; color: #2c3e50;">— {rec['title']}</span>
            </div>
            <div style="color: #555; font-size: 14px; margin-top: 4px; line-height: 1.5;">
                {rec['description']}
            </div>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# ─── Customer Profile Summary ───────────────────────────────────────────────
st.header("Customer Profile Summary")
profile_data = {
    "Attribute": ["Tenure", "Monthly Charges", "Total Charges", "Contract Type",
                  "Internet Service", "Online Security", "Tech Support", "Payment Method", "Paperless Billing"],
    "Value": [f"{tenure} months", f"${monthly_charges:.2f}", f"${total_charges:.2f}",
              contract, internet_service, online_security, tech_support, 
              payment_method, paperless_billing]
}
st.dataframe(profile_data, use_container_width=True, hide_index=True)

st.divider()

# ─── Engineered Features Expander ───────────────────────────────────────────
with st.expander("Engineered Features & Pipeline Input", expanded=False):
    st.markdown("**Features computed and sent to the predictive model:**")
    
    # Prepare engineered features data with interpretations and risk impact
    eng_data = {
        "Feature": [
            "tenure",
            "MonthlyCharges",
            "Contract",
            "ContractRisk_log",
            "ChargeTrend",
            "MonthlyToTotalRatio_log",
            "InternetService",
            "OnlineSecurity",
            "PaymentMethod",
            "PaperlessBilling",
            "TechSupport"
        ],
        "Value": [
            f"{tenure_val:.1f}",
            f"${monthly_charges_val:.2f}",
            f"{contract_encoded}",
            f"{contract_risk_log:.4f}",
            f"{charge_trend:+.4f}",
            f"{monthly_to_total_ratio_log:.4f}",
            f"{internet_encoded}",
            f"{online_security_encoded}",
            f"{payment_encoded}",
            f"{paperless_encoded}",
            f"{tech_support_encoded}"
        ],
        "Interpretation": [
            "Customer tenure in months (longer = lower risk)",
            "Current monthly bill amount",
            "Contract type (0=Month, 1=One year, 2=Two year)",
            "New + expensive = highest risk signal",
            "Bill increase from historical (+ = risk)",
            "New customer paying large bill = risk",
            "Service type (0=DSL, 1=Fiber, 2=None)",
            "Has online security protection (0=No, 1=Yes)",
            "Payment method (0=Check, 1=Card, 2=Check, 3=Auto)",
            "Receives bills digitally (0=higher engagement, 1=lower engagement)",
            "Has tech support services (0=No, 1=Yes)"
        ],
        "Risk Impact": [
            "Low",
            "Medium",
            "High",
            "High",
            "High",
            "Medium",
            "Low",
            "Low",
            "Low",
            "Low",
            "Low"
        ]
    }

    eng_df = pd.DataFrame(eng_data)

    # Styling function - highlight Contract Risk and color by impact
    def highlight_and_color(row):
        styles = []
        is_contract_risk = "ContractRisk" in row["Feature"]  

        for idx, val in enumerate(row):
            if is_contract_risk:
                if idx == len(row) - 1:  
                    styles.append("background-color: #ffe6e6; color: #c0392b; font-weight: bold;")
                else:
                    styles.append("background-color: #fef9e7; font-weight: bold;")  
            elif idx == len(row) - 1:  # Risk Impact column
                impact = row["Risk Impact"]
                if impact == "High":
                    styles.append("background-color: #ffe6e6; color: #c0392b; font-weight: bold;")
                elif impact == "Medium":
                    styles.append("background-color: #fff9e6; color: #d68910; font-weight: bold;")
                else:
                    styles.append("background-color: #e6f9e6; color: #1e8449; font-weight: bold;")
            else:
                styles.append("")
        return styles

    st.dataframe(
        eng_df.style
        .apply(highlight_and_color, axis=1)
        .set_properties(**{
            'text-align': 'left',
            'padding': '8px 12px',
        })
        .set_table_styles([
            {'selector': 'thead th', 'props': [('background-color', '#1f4e79'), ('color', 'white'), ('font-weight', 'bold')]},
            {'selector': 'tbody tr:hover', 'props': [('background-color', '#f0f2f6')]},
        ]),
        use_container_width=True,
        hide_index=True,
        height=350
    )
    
    st.divider()
    
    # Risk metrics inside expander
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="High Risk Signals",
            value=sum([
                charge_trend > 5,
                contract_risk_log > 0.5,
                monthly_to_total_ratio_log > 0.5
            ]),
            delta="Critical alerts"
        )
    with col2:
        risk_assessment = "High" if contract_risk_log > 0.5 else "Medium" if contract_risk_log > 0.2 else "Low"
        st.metric(
            label="Overall Risk",
            value=risk_assessment,
            delta="Based on Contract Risk"
        )
    with col3:
        st.metric(
            label="Churn Score",
            value=f"{churn_probability*100:.1f}%",
            delta="Model Prediction"
        )

st.divider()

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<p style='text-align: center; color: gray; font-size: 12px;'>
    Telco Churn Prediction Model | Built with Streamlit | XGBoost Model · AUC-ROC 0.847
</p>
             <div style='text-align: center; color: #888; font-size: 12px;'>
        Built by <b>Tarek Chirane  </b>
    </div>
""", unsafe_allow_html=True)
