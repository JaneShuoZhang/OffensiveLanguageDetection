import numpy as np
import os
import time
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from utils import load_train_data, load_test_data_a, RESULT_FOLDER
from preprocessing import process_train_data, process_test_data
from feature_embedding import build_ngrams_dataset, build_glove_featurized_dataset


class MLDetector:

    def __init__(self, classifier, params={}):
        classifiers_dict = {
            'LR': LogisticRegression,
            'SVC': SVC,
            'RF': RandomForestClassifier
        }
        if classifier not in classifiers_dict.keys():
            raise Exception('Available Classifiers: ', classifiers_dict.keys())
        self.classifier_name = classifier
        self.classifier = classifiers_dict[classifier]
        self.params = params
        self.model = self.classifier(**self.params)

    def fit(self, train_data, train_labels):
        return self.model.fit(train_data, train_labels)

    def predict(self, test_data):
        return self.model.predict(test_data)

    def score(self, test_data, test_labels):
        predicted_labels = self.predict(test_data)
        pos_f1 = None
        if len(set(test_labels)) == 2:
            pos_f1 = f1_score(test_labels, predicted_labels, average='binary')
        return {'mean accuracy': self.model.score(test_data, test_labels),
                'macro-f1': f1_score(test_labels, predicted_labels, average='macro'),
                'pos-f1': pos_f1}

    def hyper_tune(self, train_data, train_labels, tune_params=None, best_only=False, scoring='f1'):
        if not tune_params:
            tune_params = self.params
        tuner = GridSearchCV(self.model, tune_params, n_jobs=4, verbose=2, scoring=scoring, cv=5)
        tuner.fit(train_data, train_labels)
        self.model = tuner.best_estimator_
        if best_only:
            return {'score': tuner.best_score_,
                    'params': tuner.best_params_}
        else:
            print("Best parameter: ", tuner.best_params_)
            param_scores = {}
            results = tuner.cv_results_
            for i, param in enumerate(tuner.cv_results_['params']):
                param_str  = ', '.join("{!s}={!r}".format(key,val) for (key,val) in param.items())
                param_scores[param_str]={'train_score':results['mean_train_score'][i], 'test_score':results['mean_test_score'][i]}
            return param_scores

    def get_model(self):
        if getattr(self, 'model', None):
            return self.model
        else:
            raise Exception("Model has not been created yet.")

    def test_and_plot(self, test_data, test_labels):
        class_num = len(set(test_labels))
        predicted_labels = self.model.predict(test_data)
        macro_f1 = f1_score(test_labels, predicted_labels, average='macro')
        print("macro-f1: {}".format(macro_f1))
        pos_f1 = None
        if class_num == 2:
            pos_f1 = f1_score(test_labels, predicted_labels, average='binary')
            print("f1 for Offensive: {}".format(pos_f1))
        conf_mat = confusion_matrix(test_labels, predicted_labels)
        labels = [i for i in range(class_num)]
        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arrange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()



def prepare_data(featurizer, dim):
    if featurizer != 'ngram' and featurizer != 'glove':
        print("Please choose featurizer: 'ngram' or 'glove'.")
        return

    # Load and preprocessing data.
    train_data = load_train_data()
    train_data = process_train_data(train_data)
    test_data = load_test_data_a()
    test_data = process_test_data(test_data)

    # Get training X, y, and testing X, y
    if featurizer == 'ngram':
        train_set_ngram = build_ngrams_dataset(train_data)       
        vectorizer = train_set_ngram['vectorizer']
        test_set_ngram = build_ngrams_dataset(test_data, vectorizer=vectorizer)
    else:
        train_set_ngram = build_glove_featurized_dataset(train_data, dim)
        test_set_ngram = build_glove_featurized_dataset(test_data, dim)
    train_X = train_set_ngram['X']
    train_y = train_set_ngram['y']
    print("Shape of train_X: {}".format(train_X.shape))
    test_X = test_set_ngram['X']
    test_y = test_set_ngram['y']
    print("Shape of test_X: {}".format(test_X.shape))
    return {'train_X': train_X,
            'train_y': train_y,
            'test_X': test_X,
            'test_y': test_y}


def run_logistic_regression(featurizer, dim=300):   
    start_time = time.time()
    data = prepare_data(featurizer, dim)
    
    # Hyperparameter tuning and select best model
    lr_classifier = MLDetector('LR')
    #params_set = {'penalty':['l1'],'solver':['saga','liblinear']}
    params_set = {'penalty':['l2'],'solver':['sag','newton-cg','lbfgs']}
    lr_tune = lr_classifier.hyper_tune(data['train_X'], data['train_y'], params_set, best_only=False)
    print('Hyperparameter Tuning: ', lr_tune)

    predict_and_save(data, lr_classifier, featurizer)
    end_time = time.time()
    print("Finish logistic regression in {} mins.".format((end_time - start_time)/60))


def predict_and_save(data, classifier, featurizer):
    # Make prediction
    predictions = classifier.predict(data['test_X'])
    scores = classifier.score(data['test_X'], data['test_y'])

    # Save prediction
    origin_test_data = load_test_data_a()
    predicted_labels = ['OFF' if y==1 else 'NOT' for y in data['test_y']]
    origin_test_data['prediction'] = np.array(predicted_labels)
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    output_file_path = os.path.join(RESULT_FOLDER, "{}_{}_prediction.csv",format(classifier.classifier_name, featurizer))
    origin_test_data.to_csv(output_file_path, index=False)
    output_score_path = os.path.join(RESULT_FOLDER, "{}_{}_scores.json".format(classifier.classifier_name, featurizer))
    with open(output_score_path, 'w') as fp:
        json.dump(scores, fp)



if __name__ == "__main__":
    run_logistic_regression('glove')