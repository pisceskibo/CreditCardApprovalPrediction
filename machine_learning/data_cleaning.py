# Remove Outliers if feature is not in [Q1−3×IQR, Q3+3×IQR]
"""
Family member count
Income
Employment length
"""

# Library
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_outliers = ['Family member count', 'Income', 'Employment length']):
        self.feat_with_outliers = feat_with_outliers
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        # This feature in datasets
        if (set(self.feat_with_outliers).issubset(df.columns)):
            Q1 = df[self.feat_with_outliers].quantile(.25)      # 25% quantile
            Q3 = df[self.feat_with_outliers].quantile(.75)      # 75% quantile
            IQR = Q3 - Q1

            # Remove outliers if feature is not in [Q1−3×IQR, Q3+3×IQR]
            df = df[~((df[self.feat_with_outliers] < (Q1 - 3 * IQR)) | 
                      (df[self.feat_with_outliers] > (Q3 + 3 * IQR))).any(axis=1)]
            return df
        else:
            print("One or more features are not in the dataframe")
            return df