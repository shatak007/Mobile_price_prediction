import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("cellphone.csv")

# Define features and target
feature_columns = ['sale', 'weight', 'resolution', 'ppi', 'cpu_core', 'cpu_freq',
                   'internal_mem', 'ram', 'rear_cam', 'front_cam', 'battery', 'thickness']
X = data[feature_columns]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

# Streamlit App Setup
st.set_page_config(page_title="Mobile Phone Price Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict Price", "Compare Prices", "Data Overview", "Model Performance", "Feature Correlations"])

# Predict function
def predict_price(features_list):
    df = pd.DataFrame([features_list], columns=feature_columns)
    return model.predict(df)[0]

def predict_price_with_confidence_interval(features_list):
    prediction = predict_price(features_list)
    residuals = y_test - y_pred
    std_error = np.std(residuals)
    confidence_interval = 1.96 * std_error
    return prediction, confidence_interval

# Pages
if page == "Home":
    st.title("📱 Mobile Phone Price Prediction")
    st.markdown("Welcome! Use the sidebar to navigate the app.")

elif page == "Predict Price":
    st.title("📈 Predict a Mobile Phone's Price")
    inputs = []
    col1, col2, col3 = st.columns(3)
    with col1:
        inputs.append(st.number_input("Sale", 0, 100, 10))
        inputs.append(st.number_input("Weight (g)", 0, 500, 135))
        inputs.append(st.number_input("Resolution", 0.0, 10.0, 5.2))
        inputs.append(st.number_input("PPI", 0, 1000, 424))
    with col2:
        inputs.append(st.number_input("CPU Cores", 1, 32, 8))
        inputs.append(st.number_input("CPU Frequency (GHz)", 0.0, 5.0, 1.35))
        inputs.append(st.number_input("Internal Memory (GB)", 0, 1024, 16))
        inputs.append(st.number_input("RAM (GB)", 0, 64, 3))
    with col3:
        inputs.append(st.number_input("Rear Camera (MP)", 0, 200, 13))
        inputs.append(st.number_input("Front Camera (MP)", 0, 100, 8))
        inputs.append(st.number_input("Battery (mAh)", 0, 10000, 2160))
        inputs.append(st.number_input("Thickness (mm)", 0.0, 15.0, 7.4))

    if st.button("Predict Price"):
        price, ci = predict_price_with_confidence_interval(inputs)
        st.success(f"Predicted Price: ₹{price:.2f} ± ₹{ci:.2f}")

elif page == "Compare Prices":
    st.title("🔍 Compare Mobile Phone Prices")
    num_phones = st.slider("Number of Phones", 2, 5, 2)
    phones = []

    for i in range(num_phones):
        st.markdown(f"### 📱 Phone {i + 1}")
        col1, col2, col3 = st.columns(3)
        with col1:
            sale = st.number_input(f"Sale {i + 1}", 0, 100, 10, key=f"sale_{i}")
            weight = st.number_input(f"Weight {i + 1}", 0, 500, 135, key=f"weight_{i}")
            resolution = st.number_input(f"Resolution {i + 1}", 0.0, 10.0, 5.2, key=f"resolution_{i}")
            ppi = st.number_input(f"PPI {i + 1}", 0, 1000, 424, key=f"ppi_{i}")
        with col2:
            cpu_core = st.number_input(f"CPU Cores {i + 1}", 1, 32, 8, key=f"cpu_core_{i}")
            cpu_freq = st.number_input(f"CPU Freq {i + 1}", 0.0, 5.0, 1.35, key=f"cpu_freq_{i}")
            internal_mem = st.number_input(f"Internal Mem {i + 1}", 0, 1024, 16, key=f"internal_mem_{i}")
            ram = st.number_input(f"RAM {i + 1}", 0, 64, 3, key=f"ram_{i}")
        with col3:
            rear = st.number_input(f"Rear Cam {i + 1}", 0, 200, 13, key=f"rear_{i}")
            front = st.number_input(f"Front Cam {i + 1}", 0, 100, 8, key=f"front_{i}")
            battery = st.number_input(f"Battery {i + 1}", 0, 10000, 2160, key=f"battery_{i}")
            thick = st.number_input(f"Thickness {i + 1}", 0.0, 15.0, 7.4, key=f"thick_{i}")
        phones.append([sale, weight, resolution, ppi, cpu_core, cpu_freq, internal_mem, ram, rear, front, battery, thick])

    if st.button("Compare Prices"):
        for idx, phone in enumerate(phones):
            pred = predict_price(phone)
            st.success(f"Phone {idx+1} Predicted Price: ₹{pred:.2f}")

elif page == "Data Overview":
    st.title("📊 Dataset Overview")
    st.dataframe(data)

elif page == "Model Performance":
    st.title("📉 Model Performance")
    st.write("**Root Mean Squared Error (RMSE):**", round(rmse, 2))
    st.write("**R-squared (R²):**", round(r2, 3))

elif page == "Feature Correlations":
    st.title("📈 Feature Correlation Matrix")
    corr = data[feature_columns + ['price']].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
