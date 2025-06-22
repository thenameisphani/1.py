import pandas as pd
import numpy as np
import streamlit as st
import joblib
import pickle

model = joblib.load('xgboost_model.jb')

with open("feature_names.pkl", "rb") as file:
    feature_names = pickle.load(file)

numerical_features = ["total_sqft", "bhk", "bath"]
location_names = [f for f in feature_names if f not in numerical_features]

st.title("House Price Prediction App")
st.write("Enter the details below to predict the House Price")

sqft = st.number_input("Enter Square Feet", min_value = 500, max_value = 10000, step = 10)
bath = st.slider("Select Number of Bathrooms", 1, 10, 2)
bhk = st.slider("Select Number of BHK", 1, 10, 2)

location_input = st.text_input("Enter Location (or select below)")
location_selected = st.selectbox("Select Location", location_names)

location = location_input if location_input else location_selected

if st.button("Predict Price"):
    # Step 1: Creating an empty array of all features (set to 0)
    x_input = np.zeros(len(feature_names))

   Step 2: Assigning values to the first three features
    try:
        x_input[feature_names.index("total_sqft")] = sqft
        x_input[feature_names.index("bhk")] = bhk
        x_input[feature_names.index("bath")] = bath
    except ValueError as e:
        st.error("⚠ Feature names mismatch. Check feature_names.pkl!")
        print("Error:", e)

  Step 3: Handling one-hot encoding for the location
    if location in location_names:
        loc_index = feature_names.index(location)
        x_input[loc_index] = 1  # Set the corresponding location index to 1

  Step 4: Debugging: Checking input shape
    print("Input Vector Shape:", x_input.shape)

  Step 5: Making prediction using XGBoost model
    predicted_price = model.predict([x_input])[0]

  Step 6: Displaying the result
    st.success(f"🏠 Estimated House Price: ₹{predicted_price:,.2f} Lakhs")
