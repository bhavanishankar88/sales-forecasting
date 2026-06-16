import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sales Forecasting", layout="centered")
st.title("📈 Sales Forecasting System")
st.markdown("### Predict Future Monthly Sales")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/sales_model.pkl')

model = load_model()

col1, col2 = st.columns(2)

with col1:
    marketing_spend = st.number_input("Marketing Spend (₹)", min_value=5000, max_value=60000, value=25000, step=1000)
    num_salespeople = st.number_input("Number of Salespeople", min_value=5, max_value=30, value=12, step=1)
    holiday_season = st.selectbox("Is it Holiday Season?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

with col2:
    economic_index = st.number_input("Economic Index", min_value=80, max_value=120, value=100, step=1)
    competition_level = st.selectbox("Competition Level", [0, 1, 2], 
                                   format_func=lambda x: "Low" if x==0 else "Medium" if x==1 else "High")

if st.button("🔮 Predict Sales", type="primary"):
    input_data = pd.DataFrame({
        'marketing_spend': [marketing_spend],
        'num_salespeople': [num_salespeople],
        'holiday_season': [holiday_season],
        'economic_index': [economic_index],
        'competition_level': [competition_level]
    })
    
    prediction = model.predict(input_data)[0]
    
    st.success(f"**Predicted Monthly Sales: ₹{prediction:,.0f}**")

st.caption("Sales Forecasting Model | Built with Linear Regression")