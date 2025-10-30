import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------
st.set_page_config(page_title="🏠 House Price Prediction", layout="centered")
st.title("🏠 House Price Prediction App")
st.caption("A comparison of Simple and Multiple Linear Regression models using scikit-learn")

# ------------------------------------------
# SIDEBAR
# ------------------------------------------
model_type = st.sidebar.radio("Select Model", ["Simple Linear Regression", "Multiple Linear Regression"])

# ==========================================
# SIMPLE LINEAR REGRESSION
# ==========================================
if model_type == "Simple Linear Regression":
    st.header("📉 Simple Linear Regression (Size → Price)")

    # Load model
    try:
        model = pickle.load(open("linear_regression_model.pkl", "rb"))
        data = pd.read_csv("real_estate_price_size.csv")
    except Exception as e:
        st.error(f"❌ Error loading model or data: {e}")
        st.stop()

    st.write("This model predicts **house price** based on only one feature — the house size (sq.ft).")

    size = st.number_input("📏 Enter house size (in sq.ft)", min_value=500.0, max_value=5000.0, step=50.0)

    if st.button("🔮 Predict (Simple Model)"):
        predicted_price = float(model.predict([[size]])[0])
        st.success(f"💰 Predicted Price: **${predicted_price:,.2f}**")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(data["size"], data["price"], color="gray", alpha=0.6, label="Training Data")
        ax.scatter(size, predicted_price, color="red", s=100, label="Predicted Point")
        ax.set_xlabel("Size (sq.ft)")
        ax.set_ylabel("Price ($)")
        ax.set_title("House Price vs. Size")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        st.pyplot(fig)

# ==========================================
# MULTIPLE LINEAR REGRESSION
# ==========================================
else:
    st.header("🏡 Multiple Linear Regression (Size + Features → Price)")

    # Load model and data
    try:
        model = pickle.load(open("multiple_regression_model.pkl", "rb"))
        data = pd.read_csv("multiple_regression_data.csv")
    except Exception as e:
        st.error(f"❌ Error loading model or data: {e}")
        st.stop()

    st.write("""
    This model uses multiple features to predict house price:
    - 📏 Size (sq.ft)  
    - 🛏 Bedrooms  
    - 🛁 Bathrooms  
    - 📅 Age of House  
    - 📍 Location Score
    """)

    # Input layout
    col1, col2 = st.columns(2)
    with col1:
        size = st.number_input("🏠 Size (sq.ft)", min_value=500.0, max_value=5000.0, value=1500.0, step=50.0)
        bedrooms = st.slider("🛏 Bedrooms", 1, 5, 3)
        bathrooms = st.slider("🛁 Bathrooms", 1, 4, 2)
    with col2:
        age = st.slider("📅 Age of House (years)", 0, 40, 10)
        location_score = st.slider("📍 Location Score", 1, 10, 7)

    if st.button("🔮 Predict (Multiple Model)"):
        input_data = np.array([[size, bedrooms, bathrooms, age, location_score]])
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 Predicted Price: **${predicted_price:,.2f}**")

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(data["size"], data["price"], alpha=0.5, label="Training Data", color="gray")
        ax.scatter(size, predicted_price, color="red", s=100, label="Your Prediction")
        ax.set_xlabel("Size (sq.ft)")
        ax.set_ylabel("Price ($)")
        ax.set_title("House Price vs. Size (Multiple Model)")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        st.pyplot(fig)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and scikit-learn")
