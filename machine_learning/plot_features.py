# Libraries
import matplotlib.pyplot as plt
from yellowbrick.model_selection import FeatureImportances


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
