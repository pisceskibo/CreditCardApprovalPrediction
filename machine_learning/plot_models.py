# Libraries
import matplotlib.pyplot as plt
import scikitplot as skplt
import joblib
from pathlib import Path
from yellowbrick.model_selection import FeatureImportances
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, roc_curve, roc_auc_score
from machine_learning.training_models import y_prediction_func


# Plot the feature importance of the model
def feat_importance_plot(model_trn, model_name, X_cc_train_prep, y_cc_train_prep):
    # List of models which supported Feature Importance
    if model_name not in ['sgd', 'support_vector_machine', 'gaussian_naive_bayes', 
                          'k_nearest_neighbors', 'bagging', 'neural_network']:
        # Font size of xtick and ytick 
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12

        # Top 10 most predictive features
        top_10_feat = FeatureImportances(model_trn, relative=False, topn=10)
        # Top 10 least predictive features
        bottom_10_feat = FeatureImportances(model_trn, relative=False, topn=-10)
        
        # Fit to get the feature importance for most predictive features
        plt.figure(figsize=(10, 4))
        plt.xlabel('xlabel', fontsize=14)
        top_10_feat.fit(X_cc_train_prep, y_cc_train_prep)
        top_10_feat.show()
        print('\n')
        
        # Fit to get the feature importance for least predictive features
        plt.figure(figsize=(10, 4))
        plt.xlabel('xlabel', fontsize=14)
        bottom_10_feat.fit(X_cc_train_prep, y_cc_train_prep)
        bottom_10_feat.show()
        print('\n')
    else:
        print('No feature importance for {0}'.format(model_name))
        print('\n')


# Plot the Confusion Matrix
def confusion_matrix_func(model_trn, model_name, y_cc_train_prep, final_model=False):
    if final_model == False:
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        conf_matrix = ConfusionMatrixDisplay.from_predictions(y_cc_train_prep,
                                                              y_prediction_func(model_trn, model_name),
                                                              ax=ax, cmap='Blues', values_format='d')
        # Show the plot
        plt.grid(visible=None)
        plt.xlabel('Predicted label', fontsize=14)
        plt.ylabel('True label', fontsize=14)
        plt.title('Confusion Matrix', fontsize=14)
        plt.show()
        print('\n')
    else:
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8,8))
        conf_matrix_final = ConfusionMatrixDisplay.from_predictions(y_cc_train_prep,
                                                                    y_prediction_func(model_trn, model_name, final_model=True),
                                                                    ax=ax, cmap='Blues', values_format='d')
        # Show the plot
        plt.grid(visible=None)
        plt.xlabel('Predicted label', fontsize=14)
        plt.ylabel('True label', fontsize=14)
        plt.title('Confusion Matrix', fontsize=14)
        plt.show()
        print('\n')


# Plot ROC Curve
def roc_curve_func(model_trn, model_name, X_cc_train_prep, y_cc_train_prep, final_model=False):
    if final_model == False:
        # Check the y probabilities file exists
        y_proba_path = Path('saved_models/{0}/y_cc_train_proba_{0}.sav'.format(model_name))
        try:
            y_proba_path.resolve(strict=True)
        except FileNotFoundError:
            y_cc_train_proba = model_trn.predict_proba(X_cc_train_prep)
            joblib.dump(y_cc_train_proba, y_proba_path)
        else:
            y_cc_train_proba = joblib.load(y_proba_path)

        # Plot the roc curve
        skplt.metrics.plot_roc(y_cc_train_prep, y_cc_train_proba, 
                               title = 'ROC curve for {0}'.format(model_name), 
                               cmap='cool', figsize=(8,6), text_fontsize='large')
        plt.grid(visible=None)
        plt.show()
        print('\n')
    else:
        # Check the y probabilities file exists
        y_proba_path_final = Path('saved_models_final/{0}/y_cc_train_proba_{0}_final.sav'.format(model_name))
        try:
            y_proba_path_final.resolve(strict=True)
        except FileNotFoundError:
            y_cc_train_proba_final = model_trn.predict_proba(X_cc_train_prep)
            joblib.dump(y_cc_train_proba_final,y_proba_path_final)
        else:
            y_cc_train_proba_final = joblib.load(y_proba_path_final)
        
        # Plot the roc curve
        skplt.metrics.plot_roc(y_cc_train_prep, y_cc_train_proba_final, 
                               title = 'ROC curve for {0}'.format(model_name), 
                               cmap='cool', figsize=(8,6), text_fontsize='large')
        plt.grid(visible=None)
        plt.show()
        print('\n')