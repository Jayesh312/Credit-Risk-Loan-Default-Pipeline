import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Underwriting Workbench",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS for metric cards
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. MODEL LOADING (CACHED FOR SPEED)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_production_model():
    model_path = 'model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

pipeline = load_production_model()

if not pipeline:
    st.error("🚨 Critical Error: `model.pkl` not found. Please train and save your model before launching the portal.")
    st.stop()

# Constant configuration
DECISION_THRESHOLD = 0.0702  # Optimized threshold
LGD = 0.65  # Loss Given Default (Industry average assumption: 65%)

# -----------------------------------------------------------------------------
# 3. UI LAYOUT & TABS
# -----------------------------------------------------------------------------
st.title("🏦 Intelligent Underwriting Workbench")
st.markdown("Real-time credit risk assessment, stress testing, and batch portfolio inference.")

tab_evaluate, tab_scenario, tab_batch = st.tabs([
    "📝 Single Applicant Evaluation", 
    "🧪 What-If Stress Testing", 
    "📁 Batch Portfolio Processing"
])

# =============================================================================
# TAB 1: SINGLE APPLICANT EVALUATION
# =============================================================================
with tab_evaluate:
    st.header("Applicant Profile Entry")
    
    with st.form("underwriting_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Financial Standing")
            annual_income = st.number_input("Annual Income ($)", value=65000, step=5000)
            debt_to_income = st.number_input("Debt-to-Income (%)", value=22.5, step=1.0)
            emp_length = st.number_input("Employment (Years)", value=5, min_value=0, max_value=40)
            homeownership = st.selectbox("Housing Status", ['MORTGAGE', 'RENT', 'OWN'])
            verified_income = st.selectbox("Income Verification", ['Verified', 'Source Verified', 'Not Verified'])

        with col2:
            st.subheader("Credit History")
            credit_history_years = st.number_input("Credit History (Years)", value=10, min_value=1)
            total_credit_lines = st.number_input("Total Credit Lines", value=15, min_value=1)
            open_credit_lines = st.number_input("Open Credit Lines", value=6, min_value=1)
            delinq_2y = st.number_input("Delinquencies (Last 24M)", value=0, min_value=0)
            inquiries_last_12m = st.number_input("Recent Inquiries (12M)", value=1, min_value=0)
            
        with col3:
            st.subheader("Loan Request details")
            loan_amount = st.number_input("Loan Amount ($)", value=12000, step=1000)
            term = st.selectbox("Term Length", [36, 60])
            interest_rate = st.number_input("Proposed Interest Rate (%)", value=11.5, step=0.1)
            installment = st.number_input("Monthly Installment ($)", value=395, step=10)
            loan_purpose = st.selectbox("Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other'])
            
        submit_eval = st.form_submit_button("Run Risk Assessment Engine", type="primary")

    if submit_eval:
        # Base dictionary built to match pipeline expectation
        eval_data = {
            'annual_income': annual_income, 'debt_to_income': debt_to_income, 'emp_length': emp_length,
            'credit_history_years': credit_history_years, 'total_credit_lines': total_credit_lines,
            'open_credit_lines': open_credit_lines, 'delinq_2y': delinq_2y, 'inquiries_last_12m': inquiries_last_12m,
            'loan_amount': loan_amount, 'term': term, 'interest_rate': interest_rate, 'installment': installment,
            'homeownership': homeownership, 'verified_income': verified_income, 'loan_purpose': loan_purpose,
            
            # Imputed defaults for remaining required features
            'months_since_last_delinq': np.nan, 'total_credit_limit': annual_income * 1.5,
            'total_credit_utilized': annual_income * 0.4, 'num_collections_last_12m': 0,
            'num_historical_failed_to_pay': 0, 'months_since_90d_late': np.nan, 'current_accounts_delinq': 0,
            'total_collection_amount_ever': 0, 'current_installment_accounts': 1, 'accounts_opened_24m': 2,
            'months_since_last_credit_inquiry': 6, 'num_satisfactory_accounts': open_credit_lines,
            'num_accounts_120d_past_due': 0, 'num_accounts_30d_past_due': 0, 'num_active_debit_accounts': 2,
            'total_debit_limit': 10000, 'num_total_cc_accounts': 5, 'num_open_cc_accounts': 3,
            'num_cc_carrying_balance': 2, 'num_mort_accounts': 1 if homeownership == 'MORTGAGE' else 0,
            'account_never_delinq_percent': 100.0 if delinq_2y == 0 else 90.0, 'tax_liens': 0, 
            'public_record_bankrupt': 0, 'state': 'NY', 'application_type': 'individual', 
            'grade': 'B', 'initial_listing_status': 'whole', 'disbursement_method': 'Cash'
        }
        
        df_input = pd.DataFrame([eval_data])
        
        # Inference
        pd_score = pipeline.predict_proba(df_input)[0][1]  # Probability of Default
        expected_loss = pd_score * LGD * loan_amount
        
        st.markdown("---")
        st.subheader("Assessment Results")
        
        # Store results in session state for the "What-If" tab
        st.session_state['base_applicant'] = eval_data
        st.session_state['base_pd'] = pd_score
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Probability of Default (PD)", f"{pd_score:.2%}")
        c2.metric("Loss Given Default (LGD)", f"{LGD:.0%}")
        c3.metric("Expected Loss (EL)", f"${expected_loss:,.2f}")
        
        if pd_score >= DECISION_THRESHOLD:
            c4.error("Decision: DECLINE / REVIEW")
        else:
            c4.success("Decision: APPROVED")

# =============================================================================
# TAB 2: WHAT-IF STRESS TESTING
# =============================================================================
with tab_scenario:
    st.header("🧪 Scenario Simulator")
    st.markdown("Adjust key variables dynamically to see how the applicant's risk profile responds.")
    
    if 'base_applicant' not in st.session_state:
        st.info("👈 Please evaluate an applicant in the first tab to unlock the What-If Simulator.")
    else:
        base_data = st.session_state['base_applicant'].copy()
        base_pd = st.session_state['base_pd']
        
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            st.markdown("### Adjust Parameters")
            sim_income = st.slider("Simulate Income ($)", 
                                   min_value=int(base_data['annual_income'] * 0.5), 
                                   max_value=int(base_data['annual_income'] * 2.0), 
                                   value=int(base_data['annual_income']), step=1000)
            
            sim_loan_amt = st.slider("Simulate Loan Amount ($)", 
                                     min_value=1000, 
                                     max_value=int(base_data['loan_amount'] * 2.0), 
                                     value=int(base_data['loan_amount']), step=500)
            
            sim_interest = st.slider("Simulate Interest Rate (%)", 
                                     min_value=4.0, max_value=30.0, 
                                     value=float(base_data['interest_rate']), step=0.5)
            
            base_data['annual_income'] = sim_income
            base_data['loan_amount'] = sim_loan_amt
            base_data['interest_rate'] = sim_interest
            
            # Re-predict
            df_sim = pd.DataFrame([base_data])
            sim_pd = pipeline.predict_proba(df_sim)[0][1]
            pd_delta = sim_pd - base_pd
            
        with col_s2:
            st.markdown("### Impact Analysis")
            
            m1, m2 = st.columns(2)
            m1.metric("Simulated Risk (PD)", f"{sim_pd:.2%}", delta=f"{pd_delta * 100:.2f}% shift", delta_color="inverse")
            m2.metric("New Expected Loss", f"${sim_pd * LGD * sim_loan_amt:,.2f}")
            
            # Visualizing the shift
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = sim_pd * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Simulated Default Risk (%)"},
                delta = {'reference': base_pd * 100, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                gauge = {
                    'axis': {'range': [0, 25]},
                    'bar': {'color': "rgba(0,0,0,0)"}, # hide bar, use pointer
                    'steps': [
                        {'range': [0, DECISION_THRESHOLD*100], 'color': "#d4edda"},
                        {'range': [DECISION_THRESHOLD*100, 25], 'color': "#f8d7da"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': sim_pd * 100
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 3: BATCH PORTFOLIO PROCESSING
# =============================================================================
with tab_batch:
    st.header("📁 Batch Execution Engine")
    st.markdown("Upload a raw dataset to score thousands of loans simultaneously.")
    
    uploaded_file = st.file_uploader("Upload Applicant CSV", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        
        with st.spinner("Executing Risk Engine over portfolio..."):
            # Predict probabilities
            probs = pipeline.predict_proba(batch_df)[:, 1]
            
            results_df = batch_df.copy()
            results_df['PD_Score'] = probs
            results_df['Risk_Tier'] = pd.cut(
                results_df['PD_Score'], 
                bins=[0, DECISION_THRESHOLD/2, DECISION_THRESHOLD, 1.0], 
                labels=['Low Risk', 'Medium Risk', 'High Risk']
            )
            
            st.success("✅ Batch processing complete!")
            
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                st.write(f"Total Rows Scored: **{len(results_df):,}**")
                tier_counts = results_df['Risk_Tier'].value_counts().reset_index()
                fig_tier = px.bar(tier_counts, x='Risk_Tier', y='count', color='Risk_Tier', 
                                  color_discrete_map={'Low Risk':'#28a745', 'Medium Risk':'#ffc107', 'High Risk':'#dc3545'},
                                  title="Portfolio Risk Distribution")
                st.plotly_chart(fig_tier, use_container_width=True)
            
            with col_b2:
                st.dataframe(results_df[['loan_amount', 'annual_income', 'PD_Score', 'Risk_Tier']].head(10))
                
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Annotated Portfolio",
                    data=csv,
                    file_name="portfolio_risk_scores.csv",
                    mime="text/csv",
                )
