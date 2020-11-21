import enum
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

ONE_HOT_TRAIN_PATH_70 = "../data/train_one_hot_70.csv"
ONE_HOT_TEST_PATH_70 = "../data/test_one_hot_70.csv"

#FEATURES_INDICES = [0,2,3,4,5]
FEATURES_INDICES = list(range(1,41))

def ada_boost(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    #search(train_X, train_y)
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    ADA = AdaBoostClassifier(learning_rate=0.1, n_estimators=100, base_estimator=dt)
    ADA.fit(train_X, train_y)

    pred_y = ADA.predict(test_X)
    evaluate(ADA, test_X, test_y, pred_y)

def search(train_X, train_y):
    param_depth=[5, 10, 15]
    for depth in param_depth:
        param_n_estimators = [20, 50, 80, 100]
        param_learning_rates = [0.1, 0.3, 0.5, 0.75, 1]
        param_grid = [{'n_estimators': param_n_estimators,
                       'learning_rate': param_learning_rates
                      }]
        dt = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
        gs = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=dt),
                          param_grid=param_grid,
                          scoring='accuracy')
        gs.fit(train_X, train_y)
        print("====== Stats for depth={}".format(depth))
        print("Best score from grid search: {}".format(gs.best_score_))
        print("Best parameters from grid search: {}\n".format(gs.best_params_))


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

    #train_data = pd.read_csv(TRAIN_DATA_PATH)
    #test_data = pd.read_csv(TEST_DATA_PATH)
    train_data = pd.read_csv(ONE_HOT_TRAIN_PATH_70)
    test_data = pd.read_csv(ONE_HOT_TEST_PATH_70)

    ada_boost(train_data, test_data)

    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))