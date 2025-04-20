# Libraries
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import os


# Get dictionary for classifiers
cpdef dict get_classifiers_dictionary():
    cdef list base_estimators = [
        ('decision_tree', DecisionTreeClassifier(random_state=42)),
        ('random_forest', RandomForestClassifier(random_state=42)),
        ('extra_trees', ExtraTreesClassifier(random_state=42))
    ]

    cdef dict classifiers = {
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'bagging': BaggingClassifier(random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(random_state=42),
        'extra_trees': ExtraTreesClassifier(random_state=42),
        'stacking': StackingClassifier(estimators=base_estimators, 
                                        final_estimator=LogisticRegression(random_state=42, max_iter=1000)),
        'stacking_test': StackingClassifier(estimators=base_estimators, 
                                        final_estimator=GradientBoostingClassifier(random_state=42))
    }
    return classifiers


# Save the model in folder
cpdef folder_check_model(model_name):
    if not os.path.exists('saved_ensemble_models/{}'.format(model_name)):
        os.makedirs('saved_ensemble_models/{}'.format(model_name))