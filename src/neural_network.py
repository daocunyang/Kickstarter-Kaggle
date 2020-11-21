import enum
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

TRAIN_STD_DATA_PATH = "../data/train_standardized.csv"
TEST_STD_DATA_PATH = "../data/test_standardized.csv"

DATA_PATH_1 = "../data/test_99.csv"
DATA_PATH_5 = "../data/test_95.csv"

ONE_HOT_TRAIN_PATH_70 = "../data/train_one_hot_70.csv"
ONE_HOT_TEST_PATH_70 = "../data/test_one_hot_70.csv"

#FEATURES_INDICES = [0,2,3,4,5]
FEATURES_INDICES = list(range(1,41))

def neural_network(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    #search(train_X, train_y)
    '''
    MLP = MLPClassifier(activation='relu', hidden_layer_sizes=50,
                    max_iter=500, alpha=1e-4, solver='adam', early_stopping=True,
                    tol=1e-4, random_state=1, learning_rate_init=0.1)
'''
    MLP = MLPClassifier(hidden_layer_sizes=(40, 40, 40), alpha=0.05)
    #MLP = MLPClassifier()
    MLP.fit(train_X, train_y)
    print("Ran for {} iterations".format(MLP.n_iter_))

    pred_y = MLP.predict(test_X)
    evaluate(MLP, test_X, test_y, pred_y)


def search(train_X, train_y):
    MLP = MLPClassifier()
    parameter_space = {
        'hidden_layer_sizes': [(15, 15, 15), (50, 50, 50), (80, 80, 80), (100, 100, 100)],
        #'activation': ['relu'],
        #'solver': ['adam'],
        #'learning_rate': ['constant', 'adaptive']
        'alpha': [0.0001, 0.05, 0.001, 0.01]
    }
    gs = GridSearchCV(estimator=MLP,
                      param_grid=parameter_space,
                      n_jobs=-1,
                      scoring='accuracy')
    gs.fit(train_X, train_y)
    print("Best score from grid search: {}".format(gs.best_score_))
    print("Best parameters from grid search: {}".format(gs.best_params_))

def evaluate(model, test_X, test_y, pred_y, do_print=True):
    confusion = confusion_matrix(test_y, pred_y)
    score = model.score(test_X, test_y)
    if do_print:
        print_metrics(score, confusion, test_y, pred_y)
    return score, confusion

def print_metrics(score, confusion_matrix, test_y, pred_y, feature=None):
    if feature:
        print("Best feature: {}".format(feature))
    print(confusion_matrix)
    print("test score: {}".format(score))
    print(classification_report(test_y, pred_y, target_names=['0', '1']))

if __name__ == '__main__':
    start_time = time.time()

    USE_SMALL = False
    '''
    if USE_SMALL:
        train_data = pd.read_csv(DATA_PATH_5)
        test_data = pd.read_csv(DATA_PATH_1)
    else:
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        test_data = pd.read_csv(TEST_DATA_PATH)
    '''
    # train_data = pd.read_csv(TRAIN_DATA_PATH)
    # test_data = pd.read_csv(TEST_DATA_PATH)
    train_data = pd.read_csv(ONE_HOT_TRAIN_PATH_70)
    test_data = pd.read_csv(ONE_HOT_TEST_PATH_70)

    neural_network(train_data, test_data)

    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))