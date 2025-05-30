import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the uploaded model file
model = joblib.load("insurance_model (2).pkl")

# Streamlit app configuration
st.set_page_config(page_title="Health Insurance Predictor", layout="centered")
st.title("🩺 Health Insurance Purchase Predictor")
st.markdown("Predict if a customer will buy health insurance and suggest a suitable policy.")
st.markdown("---")

# Sidebar input section
st.sidebar.header("📝 Enter Customer Health Details")

gender = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)
bmi = st.sidebar.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=50.0, value=24.5)
smoking = st.sidebar.selectbox("Smoker?", ["Yes", "No"])
chronic_disease = st.sidebar.selectbox("Has Chronic Disease?", ["Yes", "No"])
annual_income = st.sidebar.number_input("Annual Income (₹)", min_value=10000, step=10000, value=500000)
region_code = st.sidebar.number_input("Region Code", min_value=0, value=28)
policy_sales_channel = st.sidebar.number_input("Sales Channel Code", min_value=0, value=26)
vintage = st.sidebar.slider("Vintage (Days with Company)", 0, 300, 150)

# Preprocessing
gender = 1 if gender == "Male" else 0
smoking = 1 if smoking == "Yes" else 0
chronic_disease = 1 if chronic_disease == "Yes" else 0

# Combine input into model format
input_data = [[gender, age, bmi, smoking, chronic_disease, annual_income, region_code, policy_sales_channel, vintage]]

# Policy recommendation function
def recommend_policy(age, bmi, smoking, chronic_disease):
    if age < 30 and smoking == 1:
        return "🛡️ Basic Youth Health Plan"
    elif bmi >= 30 or chronic_disease == 1:
        return "🏥 Comprehensive Health Cover"
    elif age > 50:
        return "👵 Senior Citizen Health Policy"
    else:
        return "🧾 Standard Family Health Plan"

# Predict button logic
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # Probability of class 1

    st.markdown("---")
    st.subheader("📊 Prediction Result:")

    if prediction == 1:
        st.success("✅ This customer is **likely** to buy health insurance.")
        st.markdown(f"### Confidence: `{round(proba*100, 2)}%`")
        st.markdown(f"### Recommended Plan: **{recommend_policy(age, bmi, smoking, chronic_disease)}**")
    else:
        st.error("❌ This customer is **unlikely** to buy health insurance.")
        st.markdown(f"### Confidence: `{round((1-proba)*100, 2)}%`")

    # Pie chart
    fig, ax = plt.subplots()
    labels = ['Will Buy', 'Won’t Buy']
    sizes = [proba, 1 - proba]
    colors = ['#00cc44', '#ff4d4d']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')
    st.pyplot(fig)

    st.markdown("---")
    st.info("🔁 Try different inputs from the sidebar to test predictions!")

# Footer
st.markdown("""
---
👨‍💻 Developed by Saurabh Kumar Jha, Shashwat Saini & Satvik Walia | B.Tech CSE 1st Yr.
""")
