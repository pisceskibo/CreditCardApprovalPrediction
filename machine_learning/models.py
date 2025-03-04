# Libraries
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import os


# Training List of Models (77p44s)
classifiers = {
    'sgd': SGDClassifier(random_state=42, loss='perceptron'),
    'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
    'support_vector_machine': SVC(random_state=42, probability=True),           # 1 hour
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'random_forest': RandomForestClassifier(random_state=42),
    'gaussian_naive_bayes': GaussianNB(),
    'k_nearest_neighbors': KNeighborsClassifier(),
    'gradient_boosting': GradientBoostingClassifier(random_state=42),
    'linear_discriminant_analysis': LinearDiscriminantAnalysis(),
    'bagging': BaggingClassifier(random_state=42),
    'neural_network': MLPClassifier(random_state=42, max_iter=1000),            # 1/4 hours
    'adaboost': AdaBoostClassifier(random_state=42),
    'extra_trees': ExtraTreesClassifier(random_state=42),
}


# Save the model in folder
def folder_check_model(model_name):
    if not os.path.exists('saved_models/{}'.format(model_name)):
        os.makedirs('saved_models/{}'.format(model_name))
