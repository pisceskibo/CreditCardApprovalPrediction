# Library
from dataclasses import dataclass


# Object for Personal Information (Label 1)
@dataclass
class PersonalInformation:
    gender: str
    age: float
    year_of_birth: int 
    full_name: str = ""


# Object for Relationship Information (Label 2)
@dataclass
class RelationshipInformation:
    material_status: str 
    family_member_count: float
    children_count: int = 0


# Object for Own Income Information (Label 3)
@dataclass
class OwnIncome:
    dwelling_type: str
    income: int
    car_ownship: str
    property_ownship: str


# Object for Experience (Label 4)
@dataclass
class ExperienceInformation:
    employment_status: str 
    employment_length: float
    education_level: str
    job_title: str = "to_be_droped"
    account_age: float = 0.00


# Object for Contact (Label 5)
@dataclass
class ContactInformation:
    work_phone: int
    phone: int
    email: int
    work_phone_number: int = 0
    phone_number: int = 0
    email_string: str = ""
    mobile_phone: int = 1


# Object for Profile Credit Card
@dataclass
class ProfileCreditCard:
    # Overview Information
    personal_information: PersonalInformation
    relationship_information: RelationshipInformation
    own_income: OwnIncome
    experience_information: ExperienceInformation
    contact_information: ContactInformation

    # Avatar Picture Name
    uploaded_file_name: str = None
    id: int = 0

    # Target Variable
    is_high_risk: int = 0

    def get_list_profile_predict(self):
        profile_to_predict = [
            self.id,
            self.personal_information.gender[:1],
            self.own_income.car_ownship[:1],
            self.own_income.property_ownship[:1],
            self.relationship_information.children_count,
            self.own_income.income,
            self.experience_information.employment_status,
            self.experience_information.education_level,
            self.relationship_information.material_status,
            self.own_income.dwelling_type,
            self.personal_information.age,
            self.experience_information.employment_length,
            self.contact_information.mobile_phone,
            self.contact_information.work_phone,
            self.contact_information.phone,
            self.contact_information.email,
            self.experience_information.job_title,
            self.relationship_information.family_member_count,
            self.experience_information.account_age,
            self.is_high_risk
        ]
        return profile_to_predict
        