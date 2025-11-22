import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="<---repo id---->/tourist-package-prediction", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Play Store App Revenue Prediction")
st.write("""
This application predicts the expected **ad revenue** of a Play Store application
based on its characteristics such as category, installs, active users, and screen time.
Please enter the app details below to get a revenue prediction.
""")

# User input
app_category = st.selectbox("MaritalStatus", ["Divorced", "Single", "Married", "Unmarried"])
free_or_paid = st.selectbox("Designation", ["AVP", "Executive","Manager","Senior Manager","VP"])
content_rating = st.selectbox("ProductPitched", ["Basic", "Deluxe", "King", "Standard","Super Deluxe"])
screentime_category = st.selectbox("Gender", ["Female", "Male"])

app_size = st.number_input("Age)", min_value=1.0, max_value=100.0, value=20.0, step=1)
price = st.number_input("CityTier", min_value=1, max_value=3, value=1, step=1)
installs = st.number_input("DurationOfPitch", min_value=1, max_value=200, value=1, step=10)
screen_time = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5, value=1, step=1)
active_users = st.number_input("NumberOfFollowups", min_value=1, max_value=10, value=1, step=1)
short_ads = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=1, step=1)
long_ads = st.number_input("NumberOfTrips", min_value=1, max_value=40, value=1, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'MaritalStatus': app_category,
    'Designation': free_or_paid,
    'ProductPitched': content_rating,
    'Gender': screentime_category,
    'Age': app_size,
    'CityTier': price,
    'DurationOfPitch': installs,
    'NumberOfPersonVisiting': screen_time,
    'NumberOfFollowups': active_users,
    'PreferredPropertyStar': short_ads,
    'NumberOfTrips': long_ads
}])

# Predict button
if st.button("Predict Package"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Prediction is : **${prediction:,.2f}")
