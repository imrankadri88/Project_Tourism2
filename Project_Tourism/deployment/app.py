import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="imrankadri/Project-Tourism", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Prediction App")
st.write("""
This application predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them.
Please enter the customer details below to get a prediction.
""")

# User input
age = st.number_input("Age", min_value=18, max_value=100, value=35, step = 1)
typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
citytier = st.selectbox("City Tier", [1, 2, 3])
durationofpitch = st.number_input("Duration of Pitch", min_value=1, max_value=200, value=15)
occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
numberoffollowups = st.number_input("Number of Followups", min_value=1, max_value=10, value=3)
productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
preferredpropertystar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
numberoftrips = st.number_input("Number of Trips", min_value=0, max_value=50, value=3)
passport = st.selectbox("Passport", ["No", "Yes"])
pitchsatisfactionscore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
owncar = st.selectbox("Own Car", ["No", "Yes"])
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=1)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthlyincome = st.number_input("Monthly Income", min_value=0, value=20000, step = 1)

passport = 1 if passport == "Yes" else 0
owncar = 1 if owncar == "Yes" else 0

# Assemble input into DataFrame
input_data = pd.DataFrame({
    'Age': [age],
    'TypeofContact': [typeofcontact],
    'CityTier': [citytier],
    'DurationOfPitch': [durationofpitch],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfPersonVisiting': [numberofpersonvisiting],
    'NumberOfFollowups': [numberoffollowups],
    'ProductPitched': [productpitched],
    'PreferredPropertyStar': [preferredpropertystar],
    'MaritalStatus': [maritalstatus],
    'NumberOfTrips': [numberoftrips],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitchsatisfactionscore],
    'OwnCar': [owncar],
    'NumberOfChildrenVisiting': [numberofchildrenvisiting],
    'Designation': [designation],
    'MonthlyIncome': [monthlyincome]
})

if st.button("Predict Now"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "Potential Customer" if prediction == 1 else "Not a Potential Customer"

    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
    st.metric("Purchase Probability", f"{probability:.1%}")
