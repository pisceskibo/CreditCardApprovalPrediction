# Libraries
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import numpy as np


# Y Prediction for models
def y_prediction_func(model_trn, model_name, X_cc_train_prep, y_cc_train_prep, final_model=False):
    if final_model == False:
        # Check if y_train_copy_pred files exist
        y_cc_train_pred_path = Path('saved_models/{0}/y_train_copy_pred_{0}.sav'.format(model_name))
        try:
            y_cc_train_pred_path.resolve(strict=True)
        except FileNotFoundError:
            y_cc_train_pred = cross_val_predict(model_trn, X_cc_train_prep, y_cc_train_prep,
                                                cv=10, n_jobs=-1)
            joblib.dump(y_cc_train_pred, y_cc_train_pred_path)
            return y_cc_train_pred
        else:
            y_cc_train_pred = joblib.load(y_cc_train_pred_path)
            return y_cc_train_pred
    else:
        # Check if y_train_copy_pred files exist
        y_cc_train_pred_path_final = Path('saved_models_final/{0}/y_train_copy_pred_{0}_final.sav'.format(model_name))
        try:
            y_cc_train_pred_path_final.resolve(strict=True)
        except FileNotFoundError:
            y_cc_train_pred_final = cross_val_predict(model_trn, X_cc_train_prep, y_cc_train_prep,
                                                      cv=10, n_jobs=-1)
            joblib.dump(y_cc_train_pred_final,y_cc_train_pred_path_final)
            return y_cc_train_pred_final
        else:
            y_cc_train_pred_final = joblib.load(y_cc_train_pred_path_final)
            return y_cc_train_pred_final


# Training Model
def train_model(model, model_name, X_cc_train_prep, y_cc_train_prep, final_model=False):
    if final_model == False:
        model_file_path = Path('saved_models/{0}/{0}_model.sav'.format(model_name))
        try:
            model_file_path.resolve(strict=True)
        except FileNotFoundError:
            if model_name == 'sgd':
                # Training Models
                calibrated_model = CalibratedClassifierCV(model, cv=10, method='sigmoid')
                model_trn = calibrated_model.fit(X_cc_train_prep, y_cc_train_prep)  
            else:
                model_trn = model.fit(X_cc_train_prep,y_cc_train_prep)
            
            # Save the model
            joblib.dump(model_trn,model_file_path)
            return model_trn
        else:
            model_trn = joblib.load(model_file_path)
            return model_trn
    else:
        # Check the final model file exist
        final_model_file_path = Path('saved_models_final/{0}/{0}_model.sav'.format(model_name))
        try:
            final_model_file_path.resolve(strict=True)
        except FileNotFoundError:
            # Training Models
            model_trn = model.fit(X_cc_train_prep, y_cc_train_prep)    
            joblib.dump(model_trn, final_model_file_path)
            return model_trn
        else:
            model_trn = joblib.load(final_model_file_path)
            return model_trn
        