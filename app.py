import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Credit Risk Assessment UI",
    page_icon="💳",
    layout="wide"
)

# Load the trained Pipeline
@st.cache_resource
def load_pipeline():
    return joblib.load("loan_default_pipeline.pkl")

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"⚠️ Could not load `loan_default_pipeline.pkl`. Ensure the file is in the same directory and pushed to GitHub. Details: {e}")
    st.stop()

st.title("💳 Credit Risk & Loan Default Assessment")
st.markdown("Fill in borrower details to predict loan default probability using the trained pipeline.")

st.markdown("---")

# Sidebar - Decision Threshold Adjustment
st.sidebar.header("⚙️ Risk Threshold Settings")
custom_threshold = st.sidebar.slider(
    "Default Decision Threshold",
    min_value=0.01,
    max_value=0.50,
    value=0.05,
    step=0.01,
    help="Due to extreme class imbalance (~1.78% base rate), standard 0.5 threshold will miss defaults. Adjust this threshold to calibrate recall."
)

# Streamlit Form with structured input tabs
with st.form("loan_prediction_form"):
    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Loan & Borrower Profile",
        "📊 Financial Ratios & Income",
        "📜 Credit History & Inquiries",
        "⚠️ Delinquency & Public Records"
    ])

    # --- TAB 1: LOAN & BORROWER PROFILE ---
    with tab1:
        st.subheader("Loan & Basic Borrower Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loan_amount = st.number_input("Loan Amount ($)", min_value=1000.0, max_value=100000.0, value=15000.0, step=500.0)
            term = st.selectbox("Term (Months)", options=[36, 60], index=0)
            interest_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=35.0, value=11.5, step=0.1)
            installment = st.number_input("Monthly Installment ($)", min_value=50.0, max_value=3000.0, value=495.0, step=10.0)

        with col2:
            grade = st.selectbox("Loan Grade", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=1)
            homeownership = st.selectbox("Homeownership Status", options=['MORTGAGE', 'RENT', 'OWN'], index=0)
            verified_income = st.selectbox("Income Verification Status", options=['Verified', 'Source Verified', 'Not Verified'], index=1)
            emp_length = st.number_input("Employment Length (Years)", min_value=0.0, max_value=50.0, value=5.0, step=1.0)

        with col3:
            loan_purpose = st.selectbox("Loan Purpose", options=[
                'debt_consolidation', 'credit_card', 'home_improvement', 
                'major_purchase', 'medical', 'small_business', 'car', 'other'
            ], index=0)
            application_type = st.selectbox("Application Type", options=['individual', 'joint'], index=0)
            initial_listing_status = st.selectbox("Initial Listing Status", options=['w', 'f'], index=0)
            disbursement_method = st.selectbox("Disbursement Method", options=['Cash', 'DirectPay'], index=0)
            state = st.selectbox("US State", options=[
                'CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA', 'OTHER'
            ], index=0)

    # --- TAB 2: FINANCIAL RATIOS & INCOME ---
    with tab2:
        st.subheader("Income & Credit Utilization")
        col1, col2 = st.columns(2)

        with col1:
            annual_income = st.number_input("Annual Income ($)", min_value=1000.0, max_value=1000000.0, value=75000.0, step=1000.0)
            debt_to_income = st.number_input("Debt-to-Income (DTI)", min_value=0.0, max_value=100.0, value=18.5, step=0.5)
            total_credit_limit = st.number_input("Total Credit Limit ($)", min_value=0.0, max_value=1000000.0, value=45000.0, step=1000.0)
            total_credit_utilized = st.number_input("Total Credit Utilized ($)", min_value=0.0, max_value=500000.0, value=12000.0, step=500.0)

        with col2:
            total_debit_limit = st.number_input("Total Debit Card Limit ($)", min_value=0.0, max_value=500000.0, value=15000.0, step=500.0)
            num_active_debit_accounts = st.number_input("Active Debit Accounts", min_value=0, max_value=50, value=3)
            num_cc_carrying_balance = st.number_input("CC Accounts Carrying Balance", min_value=0, max_value=50, value=2)

    # --- TAB 3: CREDIT HISTORY & INQUIRIES ---
    with tab3:
        st.subheader("Credit Accounts & Inquiries")
        col1, col2, col3 = st.columns(3)

        with col1:
            credit_history_years = st.number_input("Credit History (Years)", min_value=0.0, max_value=60.0, value=12.0, step=0.5)
            inquiries_last_12m = st.number_input("Inquiries (Last 12 Months)", min_value=0, max_value=30, value=1)
            months_since_last_credit_inquiry = st.number_input("Months Since Last Inquiry", min_value=0.0, max_value=120.0, value=4.0, step=1.0)

        with col2:
            total_credit_lines = st.number_input("Total Credit Lines", min_value=1, max_value=100, value=20)
            open_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=100, value=10)
            num_total_cc_accounts = st.number_input("Total Credit Card Accounts", min_value=0, max_value=50, value=8)
            num_open_cc_accounts = st.number_input("Open Credit Card Accounts", min_value=0, max_value=50, value=5)

        with col3:
            num_mort_accounts = st.number_input("Mortgage Accounts", min_value=0, max_value=20, value=1)
            current_installment_accounts = st.number_input("Current Installment Accounts", min_value=0, max_value=30, value=2)
            accounts_opened_24m = st.number_input("Accounts Opened (Last 24 Months)", min_value=0, max_value=50, value=3)
            num_satisfactory_accounts = st.number_input("Satisfactory Accounts", min_value=0, max_value=100, value=18)

    # --- TAB 4: DELINQUENCY & PUBLIC RECORDS ---
    with tab4:
        st.subheader("Delinquencies & Negative Records")
        col1, col2 = st.columns(2)

        with col1:
            delinq_2y = st.number_input("Delinquencies (Last 2 Years)", min_value=0, max_value=30, value=0)
            
            # Allow toggle for missing values (represented as NaN in model pipeline)
            has_prev_delinq = st.checkbox("Has previous delinquency record?", value=False)
            months_since_last_delinq = st.number_input("Months Since Last Delinquency", min_value=0.0, max_value=200.0, value=36.0) if has_prev_delinq else np.nan

            has_90d_late = st.checkbox("Has 90+ days late record?", value=False)
            months_since_90d_late = st.number_input("Months Since 90 Days Late", min_value=0.0, max_value=200.0, value=48.0) if has_90d_late else np.nan

            num_collections_last_12m = st.number_input("Collections (Last 12 Months)", min_value=0, max_value=20, value=0)
            total_collection_amount_ever = st.number_input("Total Collection Amount Ever ($)", min_value=0.0, max_value=100000.0, value=0.0)

        with col2:
            num_historical_failed_to_pay = st.number_input("Historical Failed to Pay Counts", min_value=0, max_value=20, value=0)
            current_accounts_delinq = st.number_input("Current Accounts Delinquent", min_value=0, max_value=10, value=0)
            account_never_delinq_percent = st.number_input("Account Never Delinquent (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
            num_accounts_120d_past_due = st.number_input("Accounts 120+ Days Past Due", min_value=0, max_value=10, value=0)
            num_accounts_30d_past_due = st.number_input("Accounts 30 Days Past Due", min_value=0, max_value=10, value=0)
            tax_liens = st.number_input("Tax Liens", min_value=0, max_value=20, value=0)
            public_record_bankrupt = st.number_input("Public Record Bankruptcies", min_value=0, max_value=10, value=0)

    # Submit Button
    submit_button = st.form_submit_button("🔍 Calculate Risk & Predict Default", use_container_width=True)

# --- PREDICTION LOGIC ---
if submit_button:
    # 1. Compute dynamic engineered ratio features matching notebook rules
    loan_to_income = loan_amount / annual_income if annual_income > 0 else np.nan
    installment_to_income = (installment * 12) / annual_income if annual_income > 0 else np.nan
    revol_util_proxy = total_credit_utilized / total_credit_limit if total_credit_limit > 0 else np.nan
    credit_lines_open_ratio = open_credit_lines / total_credit_lines if total_credit_lines > 0 else np.nan

    # 2. Assemble DataFrame matching exact column order and names expected by ColumnTransformer
    input_dict = {
        'emp_length': emp_length,
        'annual_income': annual_income,
        'debt_to_income': debt_to_income,
        'delinq_2y': delinq_2y,
        'months_since_last_delinq': months_since_last_delinq,
        'credit_history_years': credit_history_years,
        'inquiries_last_12m': inquiries_last_12m,
        'total_credit_lines': total_credit_lines,
        'open_credit_lines': open_credit_lines,
        'total_credit_limit': total_credit_limit,
        'total_credit_utilized': total_credit_utilized,
        'num_collections_last_12m': num_collections_last_12m,
        'num_historical_failed_to_pay': num_historical_failed_to_pay,
        'months_since_90d_late': months_since_90d_late,
        'current_accounts_delinq': current_accounts_delinq,
        'total_collection_amount_ever': total_collection_amount_ever,
        'current_installment_accounts': current_installment_accounts,
        'accounts_opened_24m': accounts_opened_24m,
        'months_since_last_credit_inquiry': months_since_last_credit_inquiry,
        'num_satisfactory_accounts': num_satisfactory_accounts,
        'num_accounts_120d_past_due': num_accounts_120d_past_due,
        'num_accounts_30d_past_due': num_accounts_30d_past_due,
        'num_active_debit_accounts': num_active_debit_accounts,
        'total_debit_limit': total_This error happens because a piece of conversational AI text was accidentally copy-pasted directly into your Python code, and Python doesn't know how to read it. 

A `SyntaxError: unterminated string literal` means that Python saw a quote mark open a string (in this case, `'`), but the line ended before it found the closing quote mark. 

Looking at your error message, you can see exactly what happened:

```python
'total_Here is a complete, clean starter template for a **Streamlit** `app.py`...
