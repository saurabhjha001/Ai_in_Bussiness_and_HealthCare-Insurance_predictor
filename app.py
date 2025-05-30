# This is My Web app program based on Python...
# And this Program predict that interest of people are buy or not insurance and suggest insurance to people
import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# here we Load our model..!!
model = joblib.load("insurance_model.pkl")

# My project Title & Description
st.set_page_config(page_title="Insurance Predictor & Policy Recommender", layout="centered")
st.title(" Insurance Purchase Predictor")
st.markdown("#### Predict if a customer will buy insurance and suggest the best policy accordingly!")
st.markdown("---")

# Sidebar Inputs part
st.sidebar.header("Input Customer Details")

gender = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 100, 30)
driving_license = st.sidebar.selectbox("Smoker?", ["Yes", "No"])
vehicle_age = st.sidebar.selectbox("Previuos insurance", ["< 1 Year", "1-2 Year", "> 2 Years"])
vehicle_damage = st.sidebar.selectbox("Has Chronic Disease?", ["Yes", "No"])
annual_premium = st.sidebar.number_input("Annual Premium", min_value=0, value=15000)
region_code = st.sidebar.number_input("Region Code", min_value=0, value=28)
policy_sales_channel = st.sidebar.number_input("Sales Channel", min_value=0, value=26)
vintage = st.sidebar.slider("Rating (Previous Insurance)", 0, 10, 5)

# Preprocessing
gender = 1 if gender == "Male" else 0
driving_license = 1 if driving_license == "Yes" else 0
vehicle_age_map = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
vehicle_age = vehicle_age_map[vehicle_age]
vehicle_damage = 1 if vehicle_damage == "Yes" else 0

input_data = [[gender, age, driving_license, region_code, 0,
               vehicle_age, vehicle_damage, annual_premium,
               policy_sales_channel, vintage]]

# Policy Recommendation Logic
def recommend_policy(age, vehicle_age, vehicle_damage):
    if age < 25 and vehicle_damage == 1:
        return " 🛡️ Basic Youth Health Plan"
    elif vehicle_age == 2 and vehicle_damage == 1:
        return " 🏥 Comprehensive Health Cover"
    elif age > 50:
        return " 👵 Senior Citizen Health Policy"
    else:
        return " 🧾 Standard Family Health Plan"

# Insurance Price Suggestion Logic
def suggest_insurance_price(age, premium, vehicle_damage):
    base_price = premium
    if age < 25:
        base_price *= 1.2  # 20% more for young people
    elif age > 50:
        base_price *= 0.9  # 10% discount for seniors
    if vehicle_damage == 1:
        base_price *= 1.15  # add 15% risk factor for chronic disease
    return int(base_price)

# Prediction Part
if st.button(" Predict"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # Probability of class 1

    st.markdown("---")
    st.subheader(" Prediction Result:")

    if prediction == 1:
        st.success(" This customer is **likely** to buy insurance.")
        suggested_price = suggest_insurance_price(age, annual_premium, vehicle_damage)
        st.markdown(f"### 💡 Suggested Insurance Price: ₹ **{suggested_price}**")
        st.markdown(f"### 📝 Recommended Policy: **{recommend_policy(age, vehicle_age, vehicle_damage)}**")
    else:
        st.error("  This customer is **unlikely** to buy insurance.")

    # Styling Pie Chart
    fig, ax = plt.subplots()
    labels = ['Buy Insurance', 'Not Buy']
    sizes = [proba, 1 - proba]
    colors = ['#00cc44', '#ff4d4d']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.markdown("---")
    st.info(" You can try different combinations from the sidebar to see how predictions change!")

# Footer
st.markdown("""
---
Made by Saurabh Kumar Jha , Shashwat Saini, Satvik Walia  & Nitin Kumar | B.Tech CSE 1st Yr.
""")
