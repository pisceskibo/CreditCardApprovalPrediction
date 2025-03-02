# Remove Feature is useless for datasets
"""
ID
Has a mobile phone
Children count
Job title
Account age
"""

# Library
from sklearn.base import BaseEstimator, TransformerMixin


class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_to_drop = ['ID', 'Has a mobile phone', 'Children count', 'Job title', 'Account age']):
        self.feature_to_drop = feature_to_drop

    def fit(self,df):
        return self
    
    def transform(self,df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop, axis=1, inplace=True)
            return df
        else:
            print("One or more features are not in the dataframe")
            return df