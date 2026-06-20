import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"  # This ensures sidebar is always expanded
)

# =============================================================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 0.8rem 0;
        border-bottom: 2px solid #eaeef2;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 18px 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s;
        margin-bottom: 10px;
    }
    .metric-card:hover {
        transform: translateY(-3px);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.95rem;
        opacity: 0.85;
        margin: 4px 0 0 0;
        font-weight: 500;
    }
    .metric-sub {
        font-size: 0.8rem;
        opacity: 0.7;
        margin: 0;
    }
    .result-box {
        background: #f8fafc;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border-left: 6px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .result-churn {
        border-left-color: #d62728;
    }
    .result-no-churn {
        border-left-color: #2ca02c;
    }
    .section-container {
        background: #f8fafc;
        border-radius: 12px;
        padding: 20px 25px;
        margin-bottom: 25px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    @media (max-width: 768px) {
        .main-header { font-size: 2rem; }
        .metric-value { font-size: 1.6rem; }
        .metric-card { padding: 12px; }
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODEL AND SCALER (CACHED)
# =============================================================================
@st.cache_resource
def load_model():
    """Load the trained SVM model and StandardScaler."""
    try:
        scaler = joblib.load('scaler.pkl')
        model = joblib.load('best_model.pkl')
        return scaler, model
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        st.stop()

scaler, model = load_model()

# =============================================================================
# LOAD DATA (IF AVAILABLE) FOR INSIGHTS
# =============================================================================
@st.cache_data
def load_data():
    if os.path.exists('customer_churn_data.csv'):
        df = pd.read_csv('customer_churn_data.csv')
        # Preprocess the data (same as training)
        df['InternetService'] = df['InternetService'].fillna('')
        # Encode churn
        df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
        # Encode Gender
        df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
        return df, "real"
    else:
        # If no CSV, generate synthetic data for visualisation only
        np.random.seed(42)
        n = 1000
        age = np.random.normal(40, 10, n).clip(18, 80).astype(int)
        gender = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
        tenure = np.random.exponential(20, n).clip(0, 72).astype(int)
        monthly_charges = np.random.normal(70, 30, n).clip(10, 200)
        # Simulate churn based on tenure and charges
        logit = -2.5 + 0.02*age + 0.1*gender - 0.15*tenure + 0.03*monthly_charges + np.random.normal(0, 0.5, n)
        prob = 1 / (1 + np.exp(-logit))
        churn = (prob > 0.5).astype(int)
        df = pd.DataFrame({
            'Age': age,
            'Gender': gender,
            'Tenure': tenure,
            'MonthlyCharges': monthly_charges.round(2),
            'Churn': churn
        })
        return df, "synthetic"

data, data_type = load_data()

# =============================================================================
# COMPUTE MODEL PERFORMANCE ON THE LOADED DATA (FOR INSIGHTS)
# =============================================================================
@st.cache_data
def compute_metrics(data):
    X = data[['Age', 'Gender', 'Tenure', 'MonthlyCharges']]
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_true = data['Churn']
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'y_pred': y_pred,
        'y_true': y_true
    }

metrics = compute_metrics(data)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def format_metric(value):
    return f"{value:.1%}"

def create_metric_card(title, value, subtitle="", color_gradient=True):
    gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)" if color_gradient else "#2c3e50"
    return f"""
    <div class="metric-card" style="background: {gradient};">
        <p class="metric-label">{title}</p>
        <h3 class="metric-value">{value}</h3>
        <p class="metric-sub">{subtitle}</p>
    </div>
    """

# =============================================================================
# SIDEBAR NAVIGATION - IMPROVED WITH DEBUG INFO
# =============================================================================

# Display a small note in the main area if sidebar is not visible
st.sidebar.markdown("### Navigation")
st.sidebar.markdown("---")

# Define page options
page_options = ["Home", "Predict Churn", "Model Insights"]

# Radio button with clean options
app_mode = st.sidebar.radio(
    "Select Page:",
    page_options,
    index=0,
    key="navigation"
)

st.sidebar.markdown("---")

# Sidebar with additional info
with st.sidebar.expander("About", expanded=False):
    st.markdown("""
    This app predicts customer churn using a trained SVM model.
    
    **Features:**
    - Age
    - Gender  
    - Tenure
    - Monthly Charges
    """)

st.sidebar.info(f"Data source: {data_type.capitalize()} dataset")

# Add a footer in sidebar
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit")

# For debugging - show current page in sidebar
st.sidebar.markdown(f"**Current Page:** {app_mode}")

# =============================================================================
# MAIN CONTENT AREA - Show sidebar toggle hint
# =============================================================================

# Add a small hint if sidebar might be collapsed
st.markdown("""
<div style="background-color: #e8f4f8; padding: 8px 15px; border-radius: 5px; margin-bottom: 15px; font-size: 0.9rem; border-left: 4px solid #1f77b4;">
    💡 <strong>Tip:</strong> Use the sidebar on the left to navigate between pages. 
    Click the arrow icon in the top-left corner if the sidebar is hidden.
</div>
""", unsafe_allow_html=True)

# =============================================================================
# PAGE: HOME
# =============================================================================
if app_mode == "Home":
    st.markdown('<div class="main-header">Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown("#### Identify customers at risk of churning and take proactive actions.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Problem Overview
        Customer churn is a critical business metric. It refers to the percentage of customers who stop using a company's service over a specific period.
        High churn rates can significantly impact revenue and growth.

        **Why predict churn?**
        - Reduce customer attrition
        - Improve customer satisfaction
        - Optimize retention strategies
        - Increase profitability
        """)
    with col2:
        st.markdown("""
        ### Solution Approach
        This app uses a **Machine Learning model** (SVM) trained on historical customer data to predict whether a customer will churn.

        **Features used:**
        - Age
        - Gender
        - Tenure (months with the company)
        - Monthly Charges

        **Model:** A tuned Support Vector Classifier with balanced class weights and SMOTE for handling class imbalance.
        """)

    if data_type == "synthetic":
        st.info("Using synthetic data for demonstration since 'customer_churn_data.csv' was not found.")

    st.markdown("---")
    st.markdown("### Dataset Snapshot")
    st.dataframe(data.head(10), use_container_width=True)

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", f"{len(data):,}")
    with col2:
        churn_rate = data['Churn'].mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    with col3:
        avg_tenure = data['Tenure'].mean()
        st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")

    # Distributions
    st.markdown("### Feature Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(data, x='Age', nbins=20, title='Age Distribution', color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(data, x='Tenure', nbins=20, title='Tenure Distribution', color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(data, x='MonthlyCharges', nbins=20, title='Monthly Charges Distribution', color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        gender_counts = data['Gender'].map({0: 'Male', 1: 'Female'}).value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        fig = px.pie(gender_counts, values='Count', names='Gender', title='Gender Distribution', color_discrete_sequence=['#d62728', '#9467bd'])
        st.plotly_chart(fig, use_container_width=True)

    # Churn analysis
    st.markdown("### Churn Analysis")
    col1, col2 = st.columns(2)
    with col1:
        churn_counts = data['Churn'].map({0: 'No Churn', 1: 'Churn'}).value_counts().reset_index()
        churn_counts.columns = ['Churn', 'Count']
        fig = px.pie(churn_counts, values='Count', names='Churn', title='Churn vs Non-Churn', hole=0.4,
                     color_discrete_sequence=['#2ca02c', '#d62728'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(data, x='Churn', y='Tenure', title='Tenure by Churn Status', color='Churn',
                     color_discrete_sequence=['#2ca02c', '#d62728'])
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(data, x='Churn', y='MonthlyCharges', title='Monthly Charges by Churn Status',
                     color='Churn', color_discrete_sequence=['#2ca02c', '#d62728'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(data, x='Churn', y='Age', title='Age by Churn Status',
                     color='Churn', color_discrete_sequence=['#2ca02c', '#d62728'])
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: PREDICT CHURN
# =============================================================================
elif app_mode == "Predict Churn":
    st.markdown('<div class="main-header">Predict Customer Churn</div>', unsafe_allow_html=True)
    st.markdown("Enter customer details below to get a churn prediction.")
    st.markdown("---")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30, help="Customer's age in years.")
            gender = st.selectbox("Gender", ['Male', 'Female'], help="Customer's gender.")
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=10, help="Number of months with the company.")
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, help="Monthly service charges.")

        with col2:
            st.markdown("### Feature Insights")
            st.info("""
            - **Age:** Younger customers may have higher churn rates.
            - **Gender:** Some patterns may exist in churn behavior.
            - **Tenure:** Longer tenure usually indicates lower churn risk.
            - **Monthly Charges:** Higher charges may increase churn likelihood.
            """)

        submit = st.form_submit_button("Predict Churn", use_container_width=True)

    if submit:
        # Convert gender: Female=1, Male=0 (as per training)
        gender_encoded = 1 if gender == 'Female' else 0

        # Prepare input array in the correct order: Age, Gender, Tenure, MonthlyCharges
        X = np.array([[age, gender_encoded, tenure, monthly_charges]])

        # Scale the input using the pre-fitted scaler
        X_scaled = scaler.transform(X)

        # Predict
        prediction = model.predict(X_scaled)[0]
        
        # Check if model has predict_proba method
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)[0][1]
        else:
            # For SVM without probability, use decision function
            if hasattr(model, 'decision_function'):
                decision = model.decision_function(X_scaled)[0]
                # Convert decision value to probability-like score using sigmoid
                proba = 1 / (1 + np.exp(-decision))
            else:
                # Fallback
                proba = float(prediction)

        # Display result
        st.markdown("---")
        st.markdown("### Prediction Result")

        if prediction == 1:
            result_class = "Churn"
            card_class = "result-churn"
            color = "#d62728"
        else:
            result_class = "No Churn"
            card_class = "result-no-churn"
            color = "#2ca02c"

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            confidence_display = f"{proba:.1%}" if proba is not None else "N/A"
            st.markdown(f"""
            <div class="result-box {card_class}" style="border-left-color: {color};">
                <h2 style="color: {color};">{result_class}</h2>
                <p style="font-size: 1.2rem;">Confidence: <strong>{confidence_display}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("View Input Details"):
            st.write(f"**Age:** {age}")
            st.write(f"**Gender:** {gender}")
            st.write(f"**Tenure:** {tenure} months")
            st.write(f"**Monthly Charges:** ${monthly_charges:.2f}")

# =============================================================================
# PAGE: MODEL INSIGHTS
# =============================================================================
elif app_mode == "Model Insights":
    st.markdown('<div class="main-header">Model Insights & Performance</div>', unsafe_allow_html=True)
    st.markdown("Understand how the model works and its evaluation metrics.")
    st.markdown("---")

    # Performance metrics
    st.markdown("### Model Performance (on available data)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(create_metric_card("Accuracy", format_metric(metrics['accuracy']), "Correct predictions"), unsafe_allow_html=True)
    with col2:
        st.markdown(create_metric_card("Precision", format_metric(metrics['precision']), "TP / (TP+FP)"), unsafe_allow_html=True)
    with col3:
        st.markdown(create_metric_card("Recall", format_metric(metrics['recall']), "TP / (TP+FN)"), unsafe_allow_html=True)
    with col4:
        st.markdown(create_metric_card("F1 Score", format_metric(metrics['f1']), "Harmonic mean"), unsafe_allow_html=True)

    if data_type == "synthetic":
        st.caption("Metrics computed on synthetic dataset. For demonstration only.")
    else:
        st.caption("Metrics computed on the loaded dataset. Actual test performance may vary.")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    labels = ['No Churn', 'Churn']
    fig = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale='Blues', title="Confusion Matrix")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("### Feature Correlation")
    # Use numeric columns only
    corr = data[['Age', 'Gender', 'Tenure', 'MonthlyCharges', 'Churn']].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Matrix")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Model details
    st.markdown("### Model Details")
    st.write(f"**Model Type:** {type(model).__name__}")
    st.write(f"**Number of Features:** 4 (Age, Gender, Tenure, MonthlyCharges)")
    st.write("**Preprocessing:** StandardScaler used to scale numeric features.")
    st.write(f"**Training Data:** {len(data)} records used for evaluation.")

    # Download report
    st.markdown("### Download Report")
    if st.button("Generate Model Report (CSV)"):
        report_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
        }
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="Download Report",
            data=csv,
            file_name="model_performance_report.csv",
            mime="text/csv"
        )
