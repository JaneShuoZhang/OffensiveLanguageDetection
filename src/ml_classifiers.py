import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


class MLDetector:

    def __init__(self, classifier, params={}):
        classifiers_dict = {
            'LR': LogisticRegression,
            'SVC': SVC,
            'RF': RandomForestClassifier
        }
        if model not in classifiers_dict.keys():
            raise Exception('Available Classifiers: ', classifiers_dict.keys())
        self.classifier = classifiers_dict[classifier]
        self.params = params
        self.model = self.classifier(**self.params)


    