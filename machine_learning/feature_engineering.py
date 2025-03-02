# Libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, OrdinalEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd


# Convert (Employment length, Age) to positive value (Days)
class TimeConversionHandler(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_days = ['Employment length', 'Age']):
        self.feat_with_days = feat_with_days

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if (set(self.feat_with_days).issubset(X.columns)):
            X[['Employment length', 'Age']] = np.abs(X[['Employment length', 'Age']])
            return X
        else:
            print("One or more features are not in the dataframe")
            return X


# Convert (Employment length) from retireness 365243 to 0 
class RetireeHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df):
        return self
    
    def transform(self, df):
        if 'Employment length' in df.columns:
            df_ret_idx = df['Employment length'][df['Employment length'] == 365243].index
            df.loc[df_ret_idx,'Employment length'] = 0
            return df
        else:
            print("Employment length is not in the dataframe")
            return df
        

# Reduce distribution skewness of (Income, Age)
class SkewnessHandler(BaseEstimator, TransformerMixin):
    def __init__(self,feat_with_skewness=['Income', 'Age']):
        self.feat_with_skewness = feat_with_skewness

    def fit(self, df):
        return self
    
    def transform(self, df):
        if (set(self.feat_with_skewness).issubset(df.columns)):
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        

# Convert (Has a work phone, Has a phone, Has an email) to binary number
class BinningNumToYN(BaseEstimator, TransformerMixin):
    def __init__(self, feat_with_num_enc = ['Has a work phone', 'Has a phone', 'Has an email']):
        self.feat_with_num_enc = feat_with_num_enc
    
    def fit(self, df):
        return self
    
    def transform(self, df):
        if (set(self.feat_with_num_enc).issubset(df.columns)):
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1:'Y', 
                                     0:'N'})
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


# One Hot Encoding
"""
Gender
Marital status
Dwelling
Employment status
Has a car
Has a property
Has a work phone
Has a phone
Has an email
"""
class OneHotWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_enc_ft = ['Gender', 'Marital status', 'Dwelling', 'Employment status', 
                                         'Has a car', 'Has a property', 'Has a work phone', 'Has a phone', 'Has an email']):
        self.one_hot_enc_ft = one_hot_enc_ft

    def fit(self, df):
        return self
    
    def transform(self, df):
        if (set(self.one_hot_enc_ft).issubset(df.columns)):
            # Function to one-hot encode the features
            def one_hot_enc(df,one_hot_enc_ft):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[one_hot_enc_ft])     # Training encoder 
                feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(one_hot_enc_ft)      # output feature names

                # change the one hot encoding array to a dataframe with the column names
                df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(),
                                                        columns=feat_names_one_hot_enc, index=df.index)
                return df
            
            # Function to hold the one-hot encoded features with the rest of the features that were not encoded
            def concat_with_rest(df,one_hot_enc_df, one_hot_enc_ft):
                # get the rest of the features that are not encoded
                rest_of_features = [ft for ft in df.columns if ft not in one_hot_enc_ft]
                # concatenate the rest of the features with the one hot encoded features
                df_concat = pd.concat([one_hot_enc_df, df[rest_of_features]], axis=1)
                return df_concat
            
            one_hot_enc_df = one_hot_enc(df, self.one_hot_enc_ft)
            full_df_one_hot_enc = concat_with_rest(df, one_hot_enc_df, self.one_hot_enc_ft)
            return full_df_one_hot_enc
        else:
            print("One or more features are not in the dataframe")
            return df


# Ordinal Encoding (Education Level)
class OrdinalFeatNames(BaseEstimator,TransformerMixin):
    def __init__(self, ordinal_enc_ft = ['Education level']):
        self.ordinal_enc_ft = ordinal_enc_ft

    def fit(self,df):
        return self
    
    def transform(self,df):
        if 'Education level' in df.columns:
            ordinal_enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = ordinal_enc.fit_transform(df[self.ordinal_enc_ft])
            return df
        else:
            print("Education level is not in the dataframe")
            return df
    

# Min-Max scaling (Age, Income, Employment length)
class MinMaxWithFeatNames(BaseEstimator, TransformerMixin):
    def __init__(self,min_max_scaler_ft = ['Age', 'Income', 'Employment length']):
        self.min_max_scaler_ft = min_max_scaler_ft

    def fit(self,df):
        return self
    
    def transform(self, df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print("One or more features are not in the dataframe")
            return df


# Change type of target variable (Is High Risk)
class ChangeToNumTarget(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df):
        return self
    
    def transform(self, df):
        if 'Is high risk' in df.columns:
            df['Is high risk'] = pd.to_numeric(df['Is high risk'])
            return df
        else:
            print("Is high risk is not in the dataframe")
            return df


# Oversample with SMOTE
class Oversample(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, df):
        return self
    
    def transform(self, df):
        if 'Is high risk' in df.columns:
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:, df.columns != 'Is high risk'],
                                                   df['Is high risk'])
            df_bal = pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal)], axis=1)
            return df_bal
        else:
            print("Is high risk is not in the dataframe")
            return df
        