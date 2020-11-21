import enum
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

DATA_PATH_1 = "../data/test_99.csv"
DATA_PATH_5 = "../data/test_95.csv"

ONE_HOT_TRAIN_PATH_70 = "../data/train_one_hot_70.csv"
ONE_HOT_TEST_PATH_70 = "../data/test_one_hot_70.csv"

#FEATURES_INDICES = [0,2,3,4,5]
FEATURES_INDICES = list(range(1,41))

def svm(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    #SVM = do_search_linear_svm(train_X, train_y)
    #SVM = LinearSVC(C=0.5)
    SVM = SVC(kernel='rbf', random_state=1, gamma=100.0, C=1.0)
    #SVM = SVC()
    #SVM = do_search_svm(train_X, train_y)
    SVM.fit(train_X, train_y)

    pred_y = SVM.predict(test_X)
    evaluate(SVM, test_X, test_y, pred_y)

def svm_using_sgd(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    #SGD = SGDClassifier(loss="hinge", penalty="l2", early_stopping=True, max_iter=100, shuffle=False, verbose=0)
    SGD = do_search_linear_svm_sgd(train_X, train_y)
    SGD.fit(train_X, train_y)
    # new_weights = SGD.coef_
    print("Learned weights: {}".format(SGD.coef_))
    print("Converged after {} iterations".format(SGD.n_iter_))

    pred_y = SGD.predict(test_X)
    evaluate(SGD, test_X, test_y, pred_y)


def do_search_svm(train_X, train_y):
    param_C = [0.1, 1.0, 10.0, 50, 100.0]
    param_gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    param_grid = [{'C': param_C,
                   'gamma': param_gamma,
                   'kernel': ['rbf', 'sigmoid', 'poly']}]
    gs = GridSearchCV(estimator=SVC(),

                      param_grid=param_grid,
                      scoring='accuracy')

    gs.fit(train_X, train_y)
    print("Best score from grid search: {}".format(gs.best_score_))
    print("Best parameters from grid search: {}".format(gs.best_params_))
    return gs.best_estimator_

def do_search_linear_svm_sgd(train_X, train_y):
    #param_loss = ['hinge']
    param_alpha = [0.0001, 0.001, 0.01, 0.1]
    #param_eta0 = [0.1, 0.5]
    param_grid = [{ #'eta0': param_eta0,
                    'alpha': param_alpha
                   #'loss': param_loss,
                  }]
    gs = GridSearchCV(estimator=SGDClassifier(early_stopping=True, loss='hinge'),
                      param_grid=param_grid,
                      scoring='accuracy')

    gs.fit(train_X, train_y)
    print("Best score from grid search: {}".format(gs.best_score_))
    print("Best parameters from grid search: {}".format(gs.best_params_))
    return gs.best_estimator_


def do_search_linear_svm(train_X, train_y):
    '''
    param_C = [0.1, 1.0, 10.0, 50, 100.0]
    param_gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
    param_grid = [{'C': param_C,
                   'gamma': param_gamma,
                   'kernel': ['rbf', 'sigmoid', 'poly']}]
    '''
    #param_loss = ['hinge', 'squared_hinge']
    param_C = [0.5, 1.0, 2.0, 5.0]
    param_grid = [{'C': param_C
                   #'loss': param_loss,
                  }]
    gs = GridSearchCV(estimator=LinearSVC(max_iter=1000),
                      param_grid=param_grid,
                      scoring='accuracy')

    gs.fit(train_X, train_y)
    print("Best score from grid search: {}".format(gs.best_score_))
    print("Best parameters from grid search: {}".format(gs.best_params_))
    return gs.best_estimator_

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

    #train_data = pd.read_csv(DATA_PATH_5)
    #test_data = pd.read_csv(DATA_PATH_1)
    train_data = pd.read_csv(ONE_HOT_TRAIN_PATH_70)
    test_data = pd.read_csv(ONE_HOT_TEST_PATH_70)

    svm(test_data, train_data)

    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))