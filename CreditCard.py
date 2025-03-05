# Libraries 
import pandas as pd
import streamlit as st
from deployment.data_analysis import data_split

# Training Data and Testing Data
train_original = pd.read_csv('datasets/train.csv')
test_original = pd.read_csv('datasets/test.csv')

# Full Data merge
full_data = pd.concat([train_original, test_original], axis=0)
full_data = full_data.sample(frac=1).reset_index(drop=True)

# New training data and testing data
train_original, test_original = data_split(full_data, 0.2)
train_copy = train_original.copy()
test_copy = test_original.copy()


# Deploy the model
from deployment import streamlit_interface

streamlit_interface.profile_application(full_data, train_copy)
