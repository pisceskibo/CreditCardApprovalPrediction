# Libraries
from sklearn.pipeline import Pipeline
from machine_learning import data_cleaning, feature_selection, feature_engineering


# Data Preprocessing = Data Cleaaning + Feature Selection + Feature Engineering
def full_pipeline(df):
    pipeline = Pipeline([
        ('outlier_remover', data_cleaning.OutlierRemover()),
        ('feature_dropper', feature_selection.DropFeatures()),
        ('time_conversion_handler', feature_engineering.TimeConversionHandler()),
        ('retiree_handler', feature_engineering.RetireeHandler()),
        ('skewness_handler', feature_engineering.SkewnessHandler()),
        ('binning_num_to_yn', feature_engineering.BinningNumToYN()),
        ('one_hot_with_feat_names', feature_engineering.OneHotWithFeatNames()),
        ('ordinal_feat_names', feature_engineering.OrdinalFeatNames()),
        ('min_max_with_feat_names', feature_engineering.MinMaxWithFeatNames()),
        ('change_to_num_target', feature_engineering.ChangeToNumTarget()),
        ('oversample', feature_engineering.Oversample())
    ])
    df_pipe_prep = pipeline.fit_transform(df)
    return df_pipe_prep
