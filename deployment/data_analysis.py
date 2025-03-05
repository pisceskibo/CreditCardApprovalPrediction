# LIbraries
from sklearn.model_selection import train_test_split
import pandas as pd


# Split data
def data_split(df, test_size):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# Caculate frequency for features
def value_cnt_norm_cal(df, feature):
    ftr_value_cnt = df[feature].value_counts()                                          # Value count for this feature
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100                 # Normalize the value counts
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)       # Merge Result
    
    # Column name
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat