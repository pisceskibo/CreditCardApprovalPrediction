# Libraries for Machine Learning
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime
import re  

# Module for Image
from PIL import Image
from deployment.draw_image_formatter import make_circle

# Module for Data Preprocessing
from deployment.profile_credit_card import PersonalInformation, RelationshipInformation, OwnIncome, ExperienceInformation, ContactInformation, ProfileCreditCard
from deployment.data_analysis import value_cnt_norm_cal
from deployment.data_preprocessing import full_pipeline


# Profile for applications in Streamlit
def profile_application(full_data, train_copy):
    st.write("""# Credit Card Approval Prediction 🏧""")
    st.write("""<hr style="border: 1px solid #ccc;">""", unsafe_allow_html=True)

    # LABEL 0:
    st.markdown(
        """
        <div style="text-align: center;">
            <u><h3>📷 Avatar Picture 📷</h3></u>
        </div>
        """,
        unsafe_allow_html=True
    )

    """
    Upload Avatar
    """
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        rounded_image = make_circle(image)

        col1, col2, col3 = st.columns([1, 2, 1])  
        with col2:
            uploaded_file_name = uploaded_file.name
            st.image(rounded_image, caption=uploaded_file_name, use_container_width=False)
    else:
        uploaded_file_name = None
        st.write("📌 Please upload your profile image")


    # LABEL 1:
    st.write("""<hr style="border: 1px dashed #ccc;">""", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <u><h3>📝 Thông tin cá nhân 📝</h3></u>
        </div>
        """,
        unsafe_allow_html=True
    )

    """
    Name input
    """
    st.write("""## Fullname""")
    fullname = st.text_input("Enter your fullname:")
    if fullname:
        if not re.fullmatch(r"[A-Za-zÀ-Ỹà-ỹ\s]+", fullname):  
            st.warning("⚠️ Vui lòng nhập tên chỉ chứa chữ cái, không bao gồm số hoặc ký tự đặc biệt!")

    """
    Gender input
    """
    st.write("""## Gender""")
    col1, col2 = st.columns(2)
    gender_options = ["Male", "Female"]
    input_gender = col1.radio("Select your gender:", gender_options, index=0, horizontal=True)
    st.write(f"**Selected Gender:** {input_gender}")

    """
    Age and Year input slider
    """
    st.write("""## Age""")
    current_year = datetime.datetime.now().year
    col1, col2 = st.columns(2)

    # Using session_state to be synchronized value
    if "year_of_birth" not in st.session_state:
        st.session_state.year_of_birth = current_year - 22
    if "input_age" not in st.session_state:
        st.session_state.input_age = 22

    # Updated function
    def update_year():
        st.session_state.year_of_birth = current_year - st.session_state.input_age
    def update_age():
        st.session_state.input_age = current_year - st.session_state.year_of_birth

    year_of_birth = col1.number_input("Enter your year of birth:", min_value=current_year - 70, max_value=current_year - 18,
                    value=st.session_state.year_of_birth, step=1, key="year_of_birth", on_change=update_age)
    age = col2.slider("Select your age:", min_value=18, max_value=70, 
                    value=st.session_state.input_age, step=1, key="input_age", on_change=update_year)
    input_age = np.negative(age*365.25)

    st.write(f"**Selected Age:** {age} years")
    st.write(f"**Year of Birth:** {year_of_birth}")
    
    """LABEL 1"""
    profile_personal_information = PersonalInformation(gender=input_gender, 
                                                       age=input_age, 
                                                       year_of_birth=year_of_birth, 
                                                       full_name=fullname)
    """LABEL 1"""


    # LABEL 2:
    st.write("""<hr style="border: 1px dashed #ccc;">""", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <u><h3>📑 Thông tin tình trạng quan hệ 📑</h3></u>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    """
    Marital status input dropdown
    """
    st.write("""## Marital status""")
    marital_status_values = list(value_cnt_norm_cal(full_data, "Marital status").index)
    marital_status_key = ["Married", "Single/not married", "Civil marriage", "Separated", "Widowed"]
    marital_status_dict = dict(zip(marital_status_key, marital_status_values))
    input_marital_status_key = st.selectbox("Select your marital status:", marital_status_key)
    input_marital_status_val = marital_status_dict.get(input_marital_status_key)

    """
    Family member count
    """
    st.write("""## Family member count""")
    fam_member_count = float(st.selectbox("Select your family member count:", [1, 2, 3, 4, 5, 6]))

    """LABEL 2"""
    profile_relationship_information = RelationshipInformation(material_status=input_marital_status_val, 
                                                               family_member_count=fam_member_count)
    """LABEL 2"""


    # LABEL 3:
    st.write("""<hr style="border: 1px dashed #ccc;">""", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <u><h3>💰 Thông tin tài sản và nguồn thu nhập 💰</h3></u>
        </div>
        """,
        unsafe_allow_html=True
    )

    """
    Dwelling type dropdown
    """
    st.write("""## Dwelling type""")
    dwelling_type_values = list(value_cnt_norm_cal(full_data, "Dwelling").index)
    dwelling_type_key = ["House / apartment", "Live with parents", "Municipal apartment", 
                         "Rented apartment", "Office apartment", "Co-op apartment"]
    dwelling_type_dict = dict(zip(dwelling_type_key, dwelling_type_values))
    input_dwelling_type_key = st.selectbox("Select the type of dwelling:", dwelling_type_key)
    input_dwelling_type_val = dwelling_type_dict.get(input_dwelling_type_key)
    
    """
    Income
    """
    st.write("""## Income""")
    input_income = int(st.text_input("Enter your income (in USD):", 0))

    """
    Ownship Information = Car ownship + Property ownship
    """
    st.write("## Ownship Information")
    col1, col2 = st.columns(2)
    with col1:
        # Car ownship input
        input_car_ownship = st.radio("Do you own a car?", ["Yes", "No"], index=0)
    with col2:
        # Property ownship input
        input_prop_ownship = st.radio("Do you own a property?", ["Yes", "No"], index=0)

    """LABEL 3"""
    profile_own_income = OwnIncome(dwelling_type=input_dwelling_type_val, 
                                   income=input_income, 
                                   car_ownship=input_car_ownship, 
                                   property_ownship=input_prop_ownship)
    """LABEL 3"""
    

    # LABEL 4:
    st.write("""<hr style="border: 1px dashed #ccc;">""", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <u><h3>🛠️ Thông tin kinh nghiệm làm việc 🛠️</h3></u>
        </div>
        """,
        unsafe_allow_html=True
    )

    """
    Employment status dropdown and Employment length
    """
    st.write("## Employment Information")
    col1, col2 = st.columns(2)
    
    with col1:
        # Employment status dropdown    
        employment_status_values = list(value_cnt_norm_cal(full_data, "Employment status").index)
        employment_status_key = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
        employment_status_dict = dict(zip(employment_status_key, employment_status_values))
        input_employment_status_key = st.selectbox("Select your employment status:", employment_status_key)
        input_employment_status_val = employment_status_dict.get(input_employment_status_key)
    with col2:
        # Employment length slider
        input_employment_length = np.negative(
            st.slider("Select your employment length:", value=6, min_value=0, max_value=30, step=1) * 365.25
        )

    """
    Education level dropdown
    """
    st.write("""## Education level""")
    col1, col2 = st.columns(2)

    with col1:
        edu_level_values = list(value_cnt_norm_cal(full_data, "Education level").index)
        edu_level_key = ["Secondary school", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"]
        edu_level_dict = dict(zip(edu_level_key, edu_level_values))
        input_edu_level_key = st.selectbox("Select your education status:", edu_level_key)
        input_edu_level_val = edu_level_dict.get(input_edu_level_key)
    with col2:
        job_title_majority = st.text_input("Job Title (Majority):")
        if job_title_majority:
            if not re.fullmatch(r"[A-Za-zÀ-Ỹà-ỹ\s]+", job_title_majority):  
                st.warning("⚠️ Không bao gồm số hoặc ký tự đặc biệt!")
        else:
            job_title_majority = "to_be_droped"

    """LABEL 4"""
    profile_experience_information = ExperienceInformation(employment_status=input_employment_status_val, 
                                                           employment_length=input_employment_length, 
                                                           education_level=input_edu_level_val, 
                                                           job_title=job_title_majority)
    """LABEL 4"""


    # LABEL 5:
    st.write("""<hr style="border: 1px dashed #ccc;">""", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center;">
            <u><h3>📠 Thông tin liên hệ 📠</h3></u>
        </div>
        """,
        unsafe_allow_html=True
    )

    """
    Work phone input
    """
    st.write("""## Work phone""")
    col1, col2 = st.columns(2)

    with col1:
        input_work_phone = st.radio("Do you have a work phone?", ["Yes", "No"], index=0)
        work_phone_dict = {"Yes": 1, "No": 0}
        work_phone_val = work_phone_dict.get(input_work_phone)
    with col2:
        if work_phone_val == 1:
            work_phone_number = st.text_input("Enter your work phone number:", "")
            if work_phone_number and not work_phone_number.isdigit():
                st.error("❌ Please enter numbers only!")
        else:
            work_phone_number = None 

    """
    Phone input
    """
    st.write("""## Phone""")
    col1, col2 = st.columns(2)

    with col1:
        input_phone = st.radio("Do you have a phone?", ["Yes", "No"], index=0)
        work_dict = {"Yes": 1, "No": 0}
        phone_val = work_dict.get(input_phone)
    with col2:
        if phone_val == 1:
            phone_val_number = st.text_input("Enter your phone number:", "")
            if phone_val_number and not phone_val_number.isdigit():
                st.error("❌ Please enter numbers only!")
        else:
            phone_val_number = None

    """
    Email input
    """
    st.write("""## Email""")
    col1, col2 = st.columns(2)

    with col1:
        input_email = st.radio("Do you have an email?", ["Yes", "No"], index=0)
        email_dict = {"Yes": 1, "No": 0}
        email_val = email_dict.get(input_email)
    with col2:
        if email_val == 1:
            email_val_string = st.text_input("Enter your email:", "")
            email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            if email_val_string and not re.match(email_pattern, email_val_string):
                st.error("❌ Please enter a valid email address!")
        else:
            email_val_string = None
    
    """LABEL 5"""
    profile_contact_information = ContactInformation(work_phone=work_phone_val, 
                                                     phone=phone_val,
                                                     email=email_val, 
                                                     work_phone_number=work_phone_number, 
                                                     phone_number=phone_val_number, 
                                                     email_string=email_val_string)
    """LABEL 5"""


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
    profile_to_predict_object = ProfileCreditCard(uploaded_file_name=uploaded_file_name,
                                            personal_information=profile_personal_information,
                                            relationship_information=profile_relationship_information,
                                            own_income=profile_own_income,
                                            experience_information=profile_experience_information,
                                            contact_information=profile_contact_information)
    profile_to_predict = profile_to_predict_object.get_list_profile_predict()

    # Set this application to dataframe
    profile_to_predict_df = pd.DataFrame([profile_to_predict], columns=train_copy.columns)

    # Merge Train Data and This Application to preprocessing
    train_copy_with_profile_to_pred = pd.concat([train_copy, profile_to_predict_df], ignore_index=True)
    train_copy_with_profile_to_pred_prep = full_pipeline(train_copy_with_profile_to_pred)

    # Get the row with the ID = 0, and drop the ID, and target variable
    profile_to_pred_prep = train_copy_with_profile_to_pred_prep[
        train_copy_with_profile_to_pred_prep["ID"] == 0
    ].drop(columns=["ID", "Is high risk"])

    """
    Button Click for Predict
    """
    if predict_bt:
        final_pred = make_prediction(profile_to_pred_prep)

        if final_pred is not None:
            if final_pred[0] == 0:
                st.success("### ✅ You have been approved for a credit card!")
                st.balloons()
            else:
                st.error("### ❌ You have not been approved for a credit card!")
        else:
            st.error("❗⚠️ Error: Unable to make a prediction.")


# Predict for this application (Gradient Boosting Classifier)
def make_prediction(profile_to_pred_prep):
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
        print("❌ Model file not found!")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
        return None