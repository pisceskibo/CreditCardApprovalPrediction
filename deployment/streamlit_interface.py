# Libraries for Machine Learning
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import re  

# Module for Data Preprocessing
from deployment.data_analysis import value_cnt_norm_cal
from deployment.data_preprocessing import full_pipeline


# Profile for applications in Streamlit
def profile_application(full_data, train_copy):
    st.write("""# Credit Card Approval Prediction""")
    st.write("""<hr style="border: 1px solid #ccc;">""", unsafe_allow_html=True)

    # Name input
    st.write("""## Fullname""")
    fullname = st.text_input("Enter your fullname:")
    if fullname:
        if not re.fullmatch(r"[A-Za-zÀ-Ỹà-ỹ\s]+", fullname):  
            st.warning("⚠️ Vui lòng nhập tên chỉ chứa chữ cái, không bao gồm số hoặc ký tự đặc biệt!")

    # Gender input
    st.write("""## Gender""")
    input_gender = st.radio("Select you gender:", ["Male", "Female"], index=0)

    # Age input slider
    st.write("""## Age""")
    input_age = np.negative(st.slider("Select your age:", value=22, min_value=18, max_value=70, step=1) * 365.25)

    # Marital status input dropdown
    st.write("""## Marital status""")
    marital_status_values = list(value_cnt_norm_cal(full_data, "Marital status").index)
    marital_status_key = ["Married", "Single/not married", "Civil marriage", "Separated", "Widowed"]
    marital_status_dict = dict(zip(marital_status_key, marital_status_values))
    input_marital_status_key = st.selectbox("Select your marital status", marital_status_key)
    input_marital_status_val = marital_status_dict.get(input_marital_status_key)

    # Family member count
    st.write("""## Family member count""")
    fam_member_count = float(st.selectbox("Select your family member count", [1, 2, 3, 4, 5, 6]))

    # Dwelling type dropdown
    st.write("""## Dwelling type""")
    dwelling_type_values = list(value_cnt_norm_cal(full_data, "Dwelling").index)
    dwelling_type_key = [
        "House / apartment",
        "Live with parents",
        "Municipal apartment ",
        "Rented apartment",
        "Office apartment",
        "Co-op apartment",
    ]
    dwelling_type_dict = dict(zip(dwelling_type_key, dwelling_type_values))
    input_dwelling_type_key = st.selectbox("Select the type of dwelling you reside in", dwelling_type_key)
    input_dwelling_type_val = dwelling_type_dict.get(input_dwelling_type_key)

    # Income
    st.write("""## Income""")
    input_income = int(st.text_input("Enter your income (in USD)", 0))

    # Employment status dropdown
    st.write("""## Employment status""")
    employment_status_values = list(value_cnt_norm_cal(full_data, "Employment status").index)
    employment_status_key = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
    employment_status_dict = dict(zip(employment_status_key, employment_status_values))
    input_employment_status_key = st.selectbox("Select your employment status", employment_status_key)
    input_employment_status_val = employment_status_dict.get(input_employment_status_key)

    # Employment length input slider
    st.write("""## Employment length""")
    input_employment_length = np.negative(
        st.slider(
            "Select your employment length", value=6, min_value=0, max_value=30, step=1
        ) * 365.25
    )

    # Education level dropdown
    st.write("""## Education level""")
    edu_level_values = list(value_cnt_norm_cal(full_data, "Education level").index)
    edu_level_key = ["Secondary school", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]
    edu_level_dict = dict(zip(edu_level_key, edu_level_values))
    input_edu_level_key = st.selectbox("Select your education status", edu_level_key)
    input_edu_level_val = edu_level_dict.get(input_edu_level_key)

    # Car ownship input
    st.write("""## Car ownship""")
    input_car_ownship = st.radio("Do you own a car?", ["Yes", "No"], index=0)

    # Property ownship input
    st.write("""## Property ownship""")
    input_prop_ownship = st.radio("Do you own a property?", ["Yes", "No"], index=0)

    # Work phone input
    st.write("""## Work phone""")
    input_work_phone = st.radio("Do you have a work phone?", ["Yes", "No"], index=0)
    work_phone_dict = {"Yes": 1, "No": 0}
    work_phone_val = work_phone_dict.get(input_work_phone)

    # Phone input
    st.write("""## Phone""")
    input_phone = st.radio("Do you have a phone?", ["Yes", "No"], index=0)
    work_dict = {"Yes": 1, "No": 0}
    phone_val = work_dict.get(input_phone)

    # Email input
    st.write("""## Email""")
    input_email = st.radio("Do you have an email?", ["Yes", "No"], index=0)
    email_dict = {"Yes": 1, "No": 0}
    email_val = email_dict.get(input_email)

    # Button
    st.write("""<hr style="border: 1px solid #ccc;">""", unsafe_allow_html=True)
    st.markdown("""
        <style>
            div.stButton > button {
                float: right;
            }
        </style>
        """, unsafe_allow_html=True)
    predict_bt = st.button("Predict")

    # list of all the input variables
    profile_to_predict = [
        0,                                  # ID
        input_gender[:1],                   # Gender
        input_car_ownship[:1],              # Car Ownership
        input_prop_ownship[:1],             # Property Ownership
        0,                                  # Children count (which will be dropped in the pipeline)
        input_income,                       # Income
        input_employment_status_val,        # Employment status
        input_edu_level_val,                # Education Level
        input_marital_status_val,           # Marital Status
        input_dwelling_type_val,            # Dwelling Type
        input_age,                          # Age
        input_employment_length,            # Employment Length
        1,                                  # Has a mobile phone (which will be dropped in the pipeline)
        work_phone_val,                     # Work Phone
        phone_val,                          # Phone
        email_val,                          # Email
        "to_be_droped",                     # Job Title (which will be dropped in the pipeline)
        fam_member_count,                   # Family Member Count
        0.00,                               # Account Age (which will be dropped in the pipeline)
        0,                                  # Target set to 0 as a placeholder
    ]

    # Set this application to dataframe
    profile_to_predict_df = pd.DataFrame([profile_to_predict], columns=train_copy.columns)

    # Merge Train Data and This Application to preprocessing
    train_copy_with_profile_to_pred = pd.concat([train_copy, profile_to_predict_df], ignore_index=True)
    train_copy_with_profile_to_pred_prep = full_pipeline(train_copy_with_profile_to_pred)

    # Get the row with the ID = 0, and drop the ID, and target variable
    profile_to_pred_prep = train_copy_with_profile_to_pred_prep[
        train_copy_with_profile_to_pred_prep["ID"] == 0
    ].drop(columns=["ID", "Is high risk"])


    # Button Click for Predict
    if predict_bt:
        final_pred = make_prediction(profile_to_pred_prep)

        if final_pred is not None:
            if final_pred[0] == 0:
                st.success("## You have been approved for a credit card")
                st.balloons()
            else:
                st.error("## Unfortunately, you have not been approved for a credit card")
        else:
            st.error("❌ Error: Unable to make a prediction.")


# Predict for this application
def make_prediction(profile_to_pred_prep):
    """
    Dự đoán kết quả từ mô hình Gradient Boosting Classifier đã lưu trên máy
    """
    try:
        # Load model in local
        model_path = "saved_models/gradient_boosting/gradient_boosting_model.sav"
        model = joblib.load(model_path)
        print("✅ Model loaded successfully!")

        # Change to dataframe
        if isinstance(profile_to_pred_prep, pd.Series):
            profile_to_pred_prep = profile_to_pred_prep.to_frame().T

        # Predict
        probabilities = model.predict_proba(profile_to_pred_prep)
        prediction = model.predict(profile_to_pred_prep)

        print("📊 Probabilities (0, 1):", probabilities)
        print("🎯 Final Prediction:", prediction)

        return prediction
    except FileNotFoundError:
        print("❌ Model file not found! Please check the path")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
        return None