import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score, precision_score, f1_score, accuracy_score,
    average_precision_score, precision_recall_curve, confusion_matrix
)
import joblib
import os

# Set page layout and configuration
st.set_page_config(
    page_title="Loan Default Risk Analytics & Prediction Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define Features from Notebook
NUMERIC_FEATURES = [
    'emp_length', 'annual_income', 'debt_to_income', 'delinq_2y',
    'months_since_last_delinq', 'credit_history_years', 'inquiries_last_12m',
    'total_credit_lines', 'open_credit_lines', 'total_credit_limit',
    'total_credit_utilized', 'num_collections_last_12m', 'num_historical_failed_to_pay',
    'months_since_90d_late', 'current_accounts_delinq', 'total_collection_amount_ever',
    'current_installment_accounts', 'accounts_opened_24m', 'months_since_last_credit_inquiry',
    'num_satisfactory_accounts', 'num_accounts_120d_past_due', 'num_accounts_30d_past_due',
    'num_active_debit_accounts', 'total_debit_limit', 'num_total_cc_accounts',
    'num_open_cc_accounts', 'num_cc_carrying_balance', 'num_mort_accounts',
    'account_never_delinq_percent', 'tax_liens', 'public_record_bankrupt',
    'loan_amount', 'term', 'interest_rate', 'installment'
]

CATEGORICAL_FEATURES = [
    'state', 'homeownership', 'verified_income', 'loan_purpose',
    'application_type', 'grade', 'initial_listing_status', 'disbursement_method'
]

# Helper function to preprocess dataset
def preprocess_df(df_raw):
    df = df_raw.copy()
    if 'issue_month' in df.columns and 'earliest_credit_line' in df.columns:
        df['issue_year'] = pd.to_datetime(df['issue_month'], format='%b-%Y', errors='coerce').dt.year
        df['issue_year'] = df['issue_year'].fillna(2018) # fallback default
        df['credit_history_years'] = df['issue_year'] - df['earliest_credit_line']
    elif 'credit_history_years' not in df.columns:
        df['credit_history_years'] = 10.0

    risk_statuses = ['Late (16-30 days)', 'Late (31-120 days)', 'In Grace Period', 'Charged Off']
    if 'loan_status' in df.columns:
        df['default_flag'] = df['loan_status'].isin(risk_statuses).astype(int)
    
    return df

# Cached Model Training Function
@st.cache_resource(show_spinner=True)
def train_and_cache_model(csv_path="loans_full_schema.csv"):
    if not os.path.exists(csv_path):
        st.error(f"Dataset file `{csv_path}` not found. Please upload or place the dataset in the project directory.")
        return None, None, None, None, None, None

    df_raw = pd.read_csv(csv_path)
    df = preprocess_df(df_raw)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['default_flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])

    # Best hyperparameters from GridSearch
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, df

# Sidebar Navigation
st.sidebar.title("💳 Loan Risk Engine")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation Menu",
    ["📊 Executive Overview", "🎯 Model Performance & PR Analysis", "🔮 Interactive Predictor", "📁 Batch Inference"]
)

# Load Model and Data
model, X_train, X_test, y_train, y_test, df = train_and_cache_model()

if model is None:
    st.warning("Please upload `loans_full_schema.csv` to proceed.")
    st.stop()

# -----------------------------------------------------------------------------
# PAGE 1: EXECUTIVE OVERVIEW & EDA
# -----------------------------------------------------------------------------
if page == "📊 Executive Overview":
    st.title("📊 Loan Default Risk Analytics Dashboard")
    st.markdown("Automated Credit Assessment Engine & Portfolio Health Analysis")
    st.markdown("---")

    # Top Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    total_loans = len(df)
    default_count = df['default_flag'].sum()
    default_rate = df['default_flag'].mean() * 100
    avg_loan_amt = df['loan_amount'].mean()

    col1.metric("Total Loans Analyzed", f"{total_loans:,}")
    col2.metric("High-Risk/Distress Cases", f"{default_count:,}")
    col3.metric("Observed Default Rate", f"{default_rate:.2f}%")
    col4.metric("Avg Loan Amount", f"${avg_loan_amt:,.2f}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Loan Status Distribution")
        status_counts = df['loan_status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig_status = px.pie(
            status_counts, values='Count', names='Status', hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with col_right:
        st.subheader("Target Flag Imbalance (`default_flag`)")
        flag_counts = df['default_flag'].value_counts().reset_index()
        flag_counts['Label'] = flag_counts['default_flag'].map({0: 'Non-Default (0)', 1: 'Distress/Default (1)'})
        fig_flag = px.bar(
            flag_counts, x='Label', y='count', color='Label',
            color_discrete_map={'Non-Default (0)': '#1f77b4', 'Distress/Default (1)': '#d62728'},
            text_auto=True
        )
        fig_flag.update_layout(showlegend=False, yaxis_title="Number of Borrowers")
        st.plotly_chart(fig_flag, use_container_width=True)

    st.subheader("Loan Amount Distribution by Risk Status")
    fig_hist = px.histogram(
        df, x="loan_amount", color="default_flag", barmode="overlay",
        labels={"default_flag": "Risk Flag"},
        color_discrete_map={0: "#2ca02c", 1: "#d62728"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 2: MODEL PERFORMANCE & PR ANALYSIS
# -----------------------------------------------------------------------------
elif page == "🎯 Model Performance & PR Analysis":
    st.title("🎯 Model Evaluation & Threshold Tuning")
    st.markdown("Precision-Recall Optimization for Highly Imbalanced Risk Datasets")
    st.markdown("---")

    y_proba = model.predict_proba(X_test)[:, 1]
    baseline_rate = y_test.mean()
    test_prauc = average_precision_score(y_test, y_proba)

    m1, m2, m3 = st.columns(3)
    m1.metric("No-Skill Baseline PR-AUC", f"{baseline_rate:.4f}")
    m2.metric("Test Set PR-AUC", f"{test_prauc:.4f}", delta=f"{test_prauc/baseline_rate:.1f}x Baseline")
    m3.metric("Test Sample Size", f"{len(y_test):,}")

    st.markdown("---")

    # Threshold Adjustment Slider
    st.sidebar.markdown("### ⚙️ Decision Threshold")
    chosen_threshold = st.sidebar.slider(
        "Probability Threshold",
        min_value=0.01, max_value=0.50, value=0.0702, step=0.005,
        help="Adjust classification threshold to trade off Precision vs. Recall."
    )

    y_pred_custom = (y_proba >= chosen_threshold).astype(int)

    rec = recall_score(y_test, y_pred_custom, zero_division=0)
    prec = precision_score(y_test, y_pred_custom, zero_division=0)
    f1 = f1_score(y_test, y_pred_custom, zero_division=0)
    acc = accuracy_score(y_test, y_pred_custom)

    st.subheader(f"Performance Metrics at Threshold = {chosen_threshold:.4f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (Sensitivity)", f"{rec:.1%}")
    c2.metric("Precision", f"{prec:.1%}")
    c3.metric("F1-Score", f"{f1:.3f}")
    c4.metric("Accuracy", f"{acc:.1%}")

    col_pr, col_cm = st.columns(2)

    with col_pr:
        st.subheader("Precision-Recall Curve")
        precision_vec, recall_vec, _ = precision_recall_curve(y_test, y_proba)
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=recall_vec, y=precision_vec, mode='lines', name=f'RF (AUC={test_prauc:.3f})', line=dict(color='firebrick', width=2)))
        fig_pr.add_trace(go.Scatter(x=[0, 1], y=[baseline_rate, baseline_rate], mode='lines', name=f'Baseline ({baseline_rate:.3f})', line=dict(dash='dash', color='gray')))
        fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", template="plotly_white")
        st.plotly_chart(fig_pr, use_container_width=True)

    with col_cm:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_custom)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=['Predicted Non-Default', 'Predicted Default'],
            y=['Actual Non-Default', 'Actual Default']
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 3: INTERACTIVE PREDICTOR
# -----------------------------------------------------------------------------
elif page == "🔮 Interactive Predictor":
    st.title("🔮 Single Applicant Risk Evaluator")
    st.markdown("Input applicant details to generate real-time default risk probability.")
    st.markdown("---")

    with st.form("loan_prediction_form"):
        st.subheader("1. Financial Profile & Income")
        f1, f2, f3, f4 = st.columns(4)
        annual_income = f1.number_input("Annual Income ($)", min_value=10000, max_value=2000000, value=75000, step=5000)
        debt_to_income = f2.number_input("Debt-to-Income (DTI %)", min_value=0.0, max_value=100.0, value=18.5, step=0.5)
        emp_length = f3.number_input("Employment Length (Years)", min_value=0, max_value=45, value=5)
        verified_income = f4.selectbox("Income Verification", ['Verified', 'Source Verified', 'Not Verified'])

        st.subheader("2. Loan Details")
        l1, l2, l3, l4 = st.columns(4)
        loan_amount = l1.number_input("Requested Loan Amount ($)", min_value=1000, max_value=100000, value=15000, step=1000)
        term = l2.selectbox("Loan Term (Months)", [36, 60])
        interest_rate = l3.number_input("Interest Rate (%)", min_value=4.0, max_value=35.0, value=12.5, step=0.25)
        installment = l4.number_input("Monthly Installment ($)", min_value=50, max_value=3000, value=350, step=25)

        l5, l6, l7, l8 = st.columns(4)
        grade = l5.selectbox("Credit Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        loan_purpose = l6.selectbox("Loan Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business', 'other'])
        homeownership = l7.selectbox("Home Ownership Status", ['MORTGAGE', 'RENT', 'OWN'])
        application_type = l8.selectbox("Application Type", ['individual', 'joint'])

        st.subheader("3. Credit History & Risk Factors")
        c1, c2, c3, c4 = st.columns(4)
        credit_history_years = c1.number_input("Credit History (Years)", min_value=1, max_value=60, value=12)
        total_credit_lines = c2.number_input("Total Credit Lines", min_value=1, max_value=100, value=20)
        open_credit_lines = c3.number_input("Open Credit Lines", min_value=0, max_value=50, value=8)
        inquiries_last_12m = c4.number_input("Inquiries (Last 12M)", min_value=0, max_value=30, value=2)

        c5, c6, c7, c8 = c1, c2, c3, c4
        delinq_2y = st.slider("Delinquencies (Last 2 Years)", 0, 10, 0)
        public_record_bankrupt = st.selectbox("Public Bankruptcies", [0, 1, 2, 3])
        tax_liens = st.selectbox("Tax Liens", [0, 1, 2, 3])

        submit = st.form_submit_button("⚡ Evaluate Risk Profile")

    if submit:
        input_data = pd.DataFrame([{
            'emp_length': emp_length,
            'annual_income': annual_income,
            'debt_to_income': debt_to_income,
            'delinq_2y': delinq_2y,
            'months_since_last_delinq': np.nan,
            'credit_history_years': credit_history_years,
            'inquiries_last_12m': inquiries_last_12m,
            'total_credit_lines': total_credit_lines,
            'open_credit_lines': open_credit_lines,
            'total_credit_limit': annual_income * 1.2,
            'total_credit_utilized': annual_income * 0.3,
            'num_collections_last_12m': 0,
            'num_historical_failed_to_pay': 0,
            'months_since_90d_late': np.nan,
            'current_accounts_delinq': 0,
            'total_collection_amount_ever': 0,
            'current_installment_accounts': 2,
            'accounts_opened_24m': 3,
            'months_since_last_credit_inquiry': 4,
            'num_satisfactory_accounts': open_credit_lines,
            'num_accounts_120d_past_due': 0,
            'num_accounts_30d_past_due': 0,
            'num_active_debit_accounts': 3,
            'total_debit_limit': 15000,
            'num_total_cc_accounts': 10,
            'num_open_cc_accounts': 5,
            'num_cc_carrying_balance': 3,
            'num_mort_accounts': 1 if homeownership == 'MORTGAGE' else 0,
            'account_never_delinq_percent': 100.0 if delinq_2y == 0 else 85.0,
            'tax_liens': tax_liens,
            'public_record_bankrupt': public_record_bankrupt,
            'loan_amount': loan_amount,
            'term': term,
            'interest_rate': interest_rate,
            'installment': installment,
            'state': 'CA',
            'homeownership': homeownership,
            'verified_income': verified_income,
            'loan_purpose': loan_purpose,
            'application_type': application_type,
            'grade': grade,
            'initial_listing_status': 'whole',
            'disbursement_method': 'DirectPay'
        }])

        prob = model.predict_proba(input_data)[0][1]
        threshold = 0.0702 # Tuned threshold from notebook

        st.markdown("---")
        st.subheader("Evaluation Results")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Predicted Default Probability", f"{prob:.2%}")
        col_res2.metric("Decision Threshold", f"{threshold:.2%}")
        
        if prob >= threshold:
            col_res3.error("⚠️ HIGH RISK / REJECT OR MANUAL REVIEW")
        else:
            col_res3.success("✅ LOW RISK / APPROVED")

        # Risk Score Gauge Plot
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Risk Score (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, threshold*100], 'color': "lightgreen"},
                    {'range': [threshold*100, 25], 'color': "yellow"},
                    {'range': [25, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 4: BATCH INFERENCE
# -----------------------------------------------------------------------------
elif page == "📁 Batch Inference":
    st.title("📁 Batch Loan Risk Scoring")
    st.markdown("Upload a CSV dataset of loan applications to generate batch predictions.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV file for batch processing", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Uploaded dataset contains **{len(batch_df)}** rows.")

        processed_batch = preprocess_df(batch_df)
        
        # Ensure missing features are filled
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            if col not in processed_batch.columns:
                processed_batch[col] = np.nan

        X_batch = processed_batch[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        
        probs = model.predict_proba(X_batch)[:, 1]
        batch_df['default_probability'] = probs
        batch_df['risk_decision'] = np.where(probs >= 0.0702, 'High Risk', 'Low Risk')

        st.subheader("Prediction Preview")
        st.dataframe(batch_df[['loan_amount', 'annual_income', 'interest_rate', 'default_probability', 'risk_decision']].head(10))

        # Download Button
        csv_download = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scored CSV",
            data=csv_download,
            file_name="scored_loan_predictions.csv",
            mime="text/csv"
        )
