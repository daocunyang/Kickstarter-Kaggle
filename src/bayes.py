import enum
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

FEATURES_INDICES = [0,2,3,4,5]

class TransformMode(enum.Enum):
   NONE = 0
    # Whether to standardize columns in FEATURES_TO_STANDARDIZE
   STANDARDIZE_SELECTED_COLUMNS = 1
    # Whether to scale all FEATURES, final range (0,20)
   SCALE_WITH_RANGE = 2
    # Only scale 'backers', 'usd_goal_real', 'duration_in_days', using default range
   SCALE_SELECTED_COLUMNS = 3

MODE = TransformMode.NONE

def gaussian_bayes(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]
    GNB = GaussianNB()
    GNB.fit(train_X, train_y)
    pred_y = GNB.predict(test_X)
    evaluate(GNB, test_X, test_y, pred_y)

def complement_bayes(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]
    CNB = ComplementNB()
    CNB.fit(train_X, train_y)
    pred_y = CNB.predict(test_X)
    evaluate(CNB, test_X, test_y, pred_y)

def evaluate(model, test_X, test_y, pred_y, do_print=True):
    confusion = confusion_matrix(test_y, pred_y)
    score = model.score(test_X, test_y)
    accuracy = accuracy_score(test_y, pred_y)
    result = precision_recall_fscore_support(test_y, pred_y)
    f1 = result[2][0]
    if do_print:
        print_metrics(score, confusion, test_y, pred_y)
    return accuracy, f1

def print_metrics(score, confusion_matrix, test_y, pred_y, feature=None):
    if feature:
        print("Best feature: {}".format(feature))
    print(confusion_matrix)
    print("test score: {}".format(score))
    print(classification_report(test_y, pred_y, target_names=['0', '1']))


if __name__ == '__main__':
    start_time = time.time()

    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    gaussian_bayes(train_data, test_data)
    #complement_bayes(train_data, test_data)

    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))