import enum
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron

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

def perceptron(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    ppn = Perceptron(penalty="l2", n_jobs=-1, early_stopping=True)
    ppn.fit(train_X, train_y)

    pred_y = ppn.predict(test_X)
    evaluate(ppn, test_X, test_y, pred_y)

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

    perceptron(train_data, test_data)

    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))