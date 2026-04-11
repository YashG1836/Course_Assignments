import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.set_page_config(
    page_title="UrbanNest Rent Predictor",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open('models/best_rf_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

model_data = load_model()
model = model_data['model']
label_encoders = model_data['label_encoders']
feature_cols = model_data['feature_cols']
categorical_cols = model_data['categorical_cols']

location_classes = list(label_encoders['location'].classes_)
city_classes = list(label_encoders['city'].classes_)
status_classes = list(label_encoders['Status'].classes_)
property_type_classes = list(label_encoders['property_type'].classes_)


st.title("UrbanNest Rent Prediction Engine")
st.markdown("Predict monthly rent for properties across **Mumbai, Pune, Delhi & Hisar**.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", city_classes)
    location = st.selectbox("Location", sorted(location_classes))
    property_type = st.selectbox("Property Type", property_type_classes)
    status = st.selectbox("Furnishing Status", status_classes)
    bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
    rooms_num = st.number_input("Number of Rooms", min_value=1, max_value=20, value=2)

with col2:
    size = st.number_input("Size (ft²)", min_value=100, max_value=50000, value=1000)
    num_bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)
    num_balconies = st.number_input("Balconies", min_value=0, max_value=10, value=1)
    is_negotiable = st.selectbox("Is Negotiable?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    security_deposit = st.number_input("Security Deposit (₹)", min_value=0, max_value=10000000, value=50000, step=5000)
    verification_days = st.number_input("Verification Days", min_value=0.0, max_value=5000.0, value=365.0, step=10.0)

latitude = st.number_input("Latitude", min_value=0.0, max_value=90.0, value=19.0, step=0.01)
longitude = st.number_input("Longitude", min_value=0.0, max_value=180.0, value=73.0, step=0.01)

st.markdown("---")

if st.button("Predict Rent 🏷️"):
    # Encode categorical inputs using saved LabelEncoders
    try:
        location_enc = label_encoders['location'].transform([location])[0]
    except ValueError:
        st.error(f"Location '{location}' was not seen during training.")
        st.stop()
    city_enc = label_encoders['city'].transform([city])[0]
    status_enc = label_encoders['Status'].transform([status])[0]
    property_type_enc = label_encoders['property_type'].transform([property_type])[0]

    # Build input DataFrame in the same column order as training
    input_dict = {
        'location': location_enc,
        'city': city_enc,
        'latitude': latitude,
        'longitude': longitude,
        'numBathrooms': num_bathrooms,
        'numBalconies': num_balconies,
        'isNegotiable': is_negotiable,
        'SecurityDeposit': security_deposit,
        'Status': status_enc,
        'Size_ft²': size,
        'BHK': bhk,
        'rooms_num': rooms_num,
        'property_type': property_type_enc,
        'verification_days': verification_days,
    }

    input_df = pd.DataFrame([input_dict])[feature_cols]

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Monthly Rent: ₹ {prediction:,.2f}")