import enum
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

TRAIN_STD_DATA_PATH = "../data/train_standardized.csv"
TEST_STD_DATA_PATH = "../data/test_standardized.csv"

ONE_HOT_TRAIN_PATH_70 = "../data/train_one_hot_70.csv"
ONE_HOT_TEST_PATH_70 = "../data/test_one_hot_70.csv"

#FEATURES_INDICES = [0,2,3,4,5]
FEATURES_INDICES = list(range(1,41))

def gradient_boost(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    #search(train_X, train_y)
    #search_xgboost(train_X, train_y)
    gd = HistGradientBoostingClassifier(loss='auto', max_bins=200, max_depth=10, max_leaf_nodes=35)

    #gd = XGBClassifier()
    gd.fit(train_X, train_y)

    pred_y = gd.predict(test_X)
    evaluate(gd, test_X, test_y, pred_y)


def search(train_X, train_y):
    param_learning_rates = [0.1, 0.3, 0.5]
    param_max_leaf_nodes = [31, 35, 40]
    param_max_bins = [150, 200, 250]
    param_max_depth = [5, 7, 10]
    #param_min_samples_leaf = [25, 30, 35]
    #param_gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    param_grid = [{#'learning_rate': param_learning_rates,
                   'max_leaf_nodes': param_max_leaf_nodes,
                   'max_bins':param_max_bins,
                   'max_depth': param_max_depth
                   #'min_samples_leaf': param_min_samples_leaf
                   }]
    gs = GridSearchCV(estimator=HistGradientBoostingClassifier(loss='binary_crossentropy', learning_rate=0.1),
                      param_grid=param_grid,
                      n_jobs=-1,
                      scoring='accuracy')

    gs.fit(train_X, train_y)
    print("Best score from grid search: {}".format(gs.best_score_))
    print("Best parameters from grid search: {}".format(gs.best_params_))

def search_xgboost(train_X, train_y):
    param_eta = [0.01, 0.1, 0.2]
    param_max_depth = [5, 8, 10]
    param_grid = [{  'eta': param_eta,
                     'max_depth': param_max_depth
    }]
    XGB = XGBClassifier()
    gs = GridSearchCV(estimator=XGB,
                      param_grid=param_grid,
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
    TP = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]
    f1 = float(TP/(TP+0.5*(FP+FN)))
    print("test score: {}".format(score))
    print("F1 score: {}".format(f1))
    print(classification_report(test_y, pred_y, target_names=['0', '1']))

if __name__ == '__main__':
    start_time = time.time()

    # train_data = pd.read_csv(TRAIN_DATA_PATH)
    # test_data = pd.read_csv(TEST_DATA_PATH)
    train_data = pd.read_csv(ONE_HOT_TRAIN_PATH_70)
    test_data = pd.read_csv(ONE_HOT_TEST_PATH_70)

    gradient_boost(train_data, test_data)

    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))