import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load Pre-trained Logistic Regression Model and Scaler
scaler = joblib.load('scaler_bankruptcy.pkl')
model = joblib.load('logistic_bankruptcy_model.pkl')

# App Config
st.set_page_config(page_title="Bankruptcy Prevention App", page_icon="⚠️")

st.title("🚨 Bankruptcy Prediction App")
st.markdown("""
Predict whether a company is at **risk of bankruptcy** based on financial factors.
""")

# Sidebar Inputs
st.sidebar.header("Input Company Financial Factors")

def user_input_features():
    ir = st.sidebar.slider('Industrial Risk', 0.0, 1.0, 0.5)
    mr = st.sidebar.slider('Management Risk', 0.0, 1.0, 0.5)
    ff = st.sidebar.slider('Financial Flexibility', 0.0, 1.0, 0.5)
    cr = st.sidebar.slider('Credibility', 0.0, 1.0, 0.5)
    cp = st.sidebar.slider('Competitiveness', 0.0, 1.0, 0.5)
    orisk = st.sidebar.slider('Operating Risk', 0.0, 1.0, 0.5)

    features = np.array([[ir, mr, ff, cr, cp, orisk]])
    return features

# Get User Input
input_data = user_input_features()

# Scale input
input_scaled = scaler.transform(input_data)

# Predict using Logistic Regression
prediction = model.predict(input_scaled)
probability = model.predict_proba(input_scaled)

# Display Result
st.subheader("Prediction Result:")

if prediction[0] == 1:
    st.error("⚠️ **High Risk of Bankruptcy**")
else:
    st.success("✅ **Low Risk of Bankruptcy**")

st.subheader("Prediction Probability:")
st.write(f"Bankruptcy Risk: {round(probability[0][1]*100, 2)}%")
st.write(f"Non-Bankruptcy: {round(probability[0][0]*100, 2)}%")

# Risk Meter Gauge
risk_score = round(probability[0][1]*100, 2)

fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = risk_score,
    title = {'text': "Bankruptcy Risk (%)"},
    gauge = {
        'axis': {'range': [0, 100]},
        'bar': {'color': "red"},
        'steps' : [
            {'range': [0, 30], 'color': "lightgreen"},
            {'range': [30, 70], 'color': "orange"},
            {'range': [70, 100], 'color': "red"}],
    }
))

st.plotly_chart(fig)

# Show Input Summary Table
feature_names = ['Industrial Risk', 'Management Risk', 'Financial Flexibility', 'Credibility', 'Competitiveness', 'Operating Risk']
input_df = pd.DataFrame(input_data, columns=feature_names)

st.subheader("📋 Input Summary:")
st.dataframe(input_df.style.highlight_max(axis=1, color="lightgreen"))

# Downloadable Report
result = {
    "Bankruptcy Risk (%)": risk_score,
    "Non-Bankruptcy (%)": round(probability[0][0]*100, 2)
}

result_df = pd.DataFrame([result])

st.download_button(
    label="📥 Download Result as CSV",
    data=result_df.to_csv(index=False),
    file_name='bankruptcy_risk_result.csv',
    mime='text/csv'
)

# Model Info
with st.expander("ℹ️ Model Details"):
    st.write("""
    - **Model:** Logistic Regression
    - **Test Accuracy:** 100%
    - **Cross-Validation Accuracy:** 99.6%
    - **Features Used:** 6 Financial Risk Factors
    """)

def user_input_features():
    ir = st.sidebar.text_input('Industrial Risk', value=0.5, type='float')
    mr = st.sidebar.text_input('Management Risk', value=0.5, type='float')
    ff = st.sidebar.text_input('Financial Flexibility', value=0.5, type='float')
    cr = st.sidebar.text_input('Credibility', value=0.5, type='float')
    cp = st.sidebar.text_input('Competitiveness', value=0.5, type='float')
    orisk = st.sidebar.text_input('Operating Risk', value=0.5, type='float')
    # Convert inputs to float and create a numpy array
    features = np.array([[float(ir), float(mr), float(ff), float(cr), float(cp), float(orisk)]])
    return features
