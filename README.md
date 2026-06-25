# Telco Customer Churn Predictor

> **Predicting which customers will leave 30 days before they do.**

---

## The Business Problem

A major telecom provider was losing **27% of its customers every year.**

Each lost customer represents approximately **$1,200 in annual revenue**  gone permanently, with no chance to intervene after the fact.

Traditional retention teams were reactive: they called customers who had *already* decided to leave, making costly win-back offers too late to matter.

**The question we set out to answer:**

> *Can we identify which customers are likely to cancel their subscription next month while there is still time to act?*

---

## What We Built

An end-to-end machine learning system that:

1. **Analyses customer behaviour** across 20+ variables (contract type, monthly charges, services used, tenure, payment method, and more)
2. **Scores every customer daily** with a churn probability from 0% to 100%
3. **Flags high-risk customers** 30 days in advance and feeds them into the CRM for targeted retention outreach
4. **Runs automatically** as a deployed API no manual intervention needed

---

## Results at a Glance

| What We Measure | Score | What It Means |
|---|---|---|
| **AUC-ROC** | **0.872** | The model is correct 87% of the time when ranking who is likely to churn |
| **Recall** | **0.79** | Out of every 10 customers who would churn, we catch **8 of them** in advance |
| **Precision** | **0.62** | Of all customers flagged as high-risk, **62% are genuine churners** |

## Monthly Impact Estimate (Based on Precision = 0.62)

If the retention team uses our model to target **350 high‑risk customers per month**:

- **Precision = 0.62** → ~**217 of those are genuine churners**
- With a realistic **40% save rate** → ~**87 customers retained per month**
- **Annual savings:** 87 × $1,200 × 12 ≈ **$1.25 million per year**

> *This is a conservative estimate, excluding upsell revenue from retained customers.*

## Why We Catch 8 Out of 10 Churners

The model was specifically tuned for **Recall**  the metric that measures how many real churners we detect.

In this business context, the cost of *missing* a churner (losing them forever) is far greater than the cost of a *false alarm* (sending a retention offer to someone who wasn't going to leave). We optimised accordingly.

---

## What Drives Churn & Key Findings

Through 7 models, feature engineering, and statistical testing, three patterns emerged consistently:

**1. Contract type is the single strongest predictor**
Customers on month-to-month contracts churn at **3× the rate** of customers on 2-year contracts. Encouraging new customers to commit to longer contracts is the highest-ROI retention action available.

**2. New customers with high bills are the most vulnerable**
Customers in their first 6 months paying more than $70/month show extremely high churn rates. They haven't built loyalty yet, and the price feels unjustified. Early proactive outreach a service review call, a loyalty discount, or a free upgrade  significantly reduces this risk.

**3. Service engagement predicts loyalty**
Customers using 7 or more of the available services (security, backup, streaming, etc.) almost never leave. Onboarding programs that introduce customers to additional services in their first 90 days create stickiness that pays for years.

---

## How to Use the Live Demo

👉 **[Open the Interactive Predictor](https://5b6t248enawp9rmk37kmks.streamlit.app)**

Enter a customer profile : contract type, monthly charge, tenure, and services and instantly see:
- Their churn probability score
- Their risk category (Low / Medium / High)
- The recommended retention action

No technical knowledge required. Built for account managers and retention teams.

---

## How It Works ( Non-Technical Overview )

```
Raw Customer Data
       ↓
   Data Cleaning          Remove errors, fix missing values, standardise categories
       ↓
Feature Engineering        Create 8 business-logic signals (e.g. "is this customer
                           paying more than their historical average?")
       ↓
   Model Training          7 machine learning algorithms tested and compared.
                           Best model: XGBoost, tuned with cross-validation.
       ↓
   Daily Scoring           Every customer gets a churn probability score (0–100%)
       ↓
   CRM Integration         High-risk customers (score > 48%) routed to retention team
```

---

## Accuracy vs. Competing Approaches

We tested 7 different machine learning models against a simple rule‑based baseline.  
The champion model was tuned specifically to maximise **Recall** — catching as many real churners as possible.

| Approach | Recall | Notes |
|---|---|---|
| Rule‑based (contract type only) | ~0.42 | Simple, but misses ~60% of churners |
| Logistic Regression (baseline) | **0.69** | Fast, interpretable, decent baseline |
| Random Forest | **0.74** | Good precision, higher recall than logistic regression |
| **Tuned XGBoost (Champion)** | **0.79** |  Best balance of recall and precision — catches 8/10 churners |
| Stacking Ensemble | 0.77 | Slightly lower recall, significantly higher compute cost |

All scores are evaluated on the original (non‑SMOTE) test set to reflect real‑world performance.

---

## Data & Privacy

- Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  7,043 anonymised customer records
- No personally identifiable information (PII) used at any point in modelling
- All customer scoring in production uses encrypted data channels
- Model does not use or store individual names, addresses, or payment details

---

## Project Files

```
├── notebook/
│   └── Telco_Churn_Pipeline.ipynb    Full analysis: cleaning → modelling → results
├── api/
│   └── main.py                        FastAPI endpoint for real-time scoring
├── app/
│   └── streamlit_app.py               Interactive demo for non-technical users
├── models/
│   └── churn_pipeline.pkl             Trained model + scaler (production-ready)
└── README.md
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data & Modelling | Python, Pandas, Scikit-learn, XGBoost, TensorFlow |
| Deployment | FastAPI, Docker |
| Demo Interface | Streamlit |
| Model Registry | joblib |

---

## Limitations & Next Steps

**Current limitations:**
- Dataset is US-based (~2014–2015). Churn patterns in other markets or time periods may differ.
- Missing signals that would improve accuracy: customer satisfaction scores (NPS), call centre interaction history, and complaint logs.

**Planned improvements:**
- Add SHAP value explanations so retention agents can see *why* each customer was flagged
- Retrain quarterly on fresh data to account for market shifts
- Integrate customer lifetime value (LTV) so the model prioritises high-value churners first

---

## Contact

Built by **Tarek Chirane** 
[LinkedIn](https://www.linkedin.com/in/tarek-chirane-2452b9260) · [Email](tareksttarek@gmail.com)

*Open to questions, collaborations, or feedback.*
