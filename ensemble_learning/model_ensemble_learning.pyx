# Libraries
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
import os


# Get dictionary for classifiers
cpdef dict get_classifiers_dictionary():
    cdef dict classifiers = {
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'bagging': BaggingClassifier(random_state=42)
    }
    return classifiers


# Save the model in folder
cpdef folder_check_model(model_name):
    if not os.path.exists('saved_ensemble_models/{}'.format(model_name)):
        os.makedirs('saved_ensemble_models/{}'.format(model_name))