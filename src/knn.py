import enum
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

TRAIN80_DATA_PATH = "../data/train_80.csv"
TEST80_DATA_PATH = "../data/test_80.csv"

TRAIN95_DATA_PATH = "../data/train_95.csv"
TEST95_DATA_PATH = "../data/test_95.csv"

TRAIN99_DATA_PATH = "../data/train_99.csv"
TEST99_DATA_PATH = "../data/test_99.csv"

STANDARDIZED_TRAIN_DATA_PATH = "../train_standardized.csv"
STANDARDIZED_TEST_DATA_PATH = "../test_standardized.csv"

SCALED_TRAIN_DATA_PATH = "../data/train_scaled.csv"
SCALED_TEST_DATA_PATH = "../data/test_scaled.csv"

ONE_HOT_TRAIN_PATH_70 = "../data/train_one_hot_70.csv"
ONE_HOT_TEST_PATH_70 = "../data/test_one_hot_70.csv"

FEATURES = ['main_category', 'backers', 'country', 'usd_goal_real', 'duration_in_days']
#FEATURES_INDICES = [0,2,3,4,5]
FEATURES_INDICES = list(range(1,41))

class TransformMode(enum.Enum):
   NONE = 0
    # Whether to standardize columns in FEATURES_TO_STANDARDIZE
   STANDARDIZE_SELECTED_COLUMNS = 1
    # Whether to scale all FEATURES, final range (0,20)
   SCALE_WITH_RANGE = 2
    # Only scale 'backers', 'usd_goal_real', 'duration_in_days', using default range
   SCALE_SELECTED_COLUMNS = 3

MODE = TransformMode.NONE

def knn(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]
    KNN = KNeighborsClassifier(n_neighbors=17, metric="braycurtis")
    KNN.fit(train_X, train_y)
    pred_y = KNN.predict(test_X)
    evaluate(KNN, test_X, test_y, pred_y)

    cmap = plt.get_cmap('Blues')
    plot_confusion_matrix(KNN, test_X, test_y, cmap=cmap)
    plt.show()


def knn_search_K(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    k_range = list(range(5,21))
    acc_scores = []
    f1_scores = []
    accuracy = None
    f1 = None

    # Search for the best K (result is K=15)
    for K in k_range:
        accuracy, f1 = run_knn(train_X, train_y, test_X, test_y, K)
        acc_scores.append(accuracy)
        f1_scores.append(f1)
        print("K = {}, accuracy score = {}, f1 score = {}".format(K, accuracy, f1))
    plot(k_range, acc_scores, f1_scores)

def run_knn(train_X, train_y, test_X, test_y, K):
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(train_X, train_y)
    pred_y = knn.predict(test_X)
    return evaluate(knn, test_X, test_y, pred_y)

def knn_auto_search(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    knn = do_search(train_X, train_y)
    knn.fit(train_X, train_y)
    pred_y = knn.predict(test_X)
    return evaluate(knn, test_X, test_y, pred_y)

# Search for best params using GridSearchCV
def do_search(train_X, train_y):
    param_n_neighbors = [3, 5, 8, 11, 15, 17]
    param_metric = ['braycurtis', 'minkowski', 'chebyshev']
    param_grid = [{'n_neighbors': param_n_neighbors,
                   'metric': param_metric}]
    gs = GridSearchCV(estimator=KNeighborsClassifier(),
                      param_grid=param_grid,
                      scoring='accuracy')
    gs.fit(train_X, train_y)
    print("Best score from grid search: {}".format(gs.best_score_))
    print("Best parameters from grid search: {}".format(gs.best_params_))
    return gs.best_estimator_

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

def plot(k_range, acc_scores, f1_scores):
    plt.plot(k_range, acc_scores, label="Accuracy")
    plt.plot(k_range, f1_scores, label="F1 Score")
    plt.xlabel("Value of K")
    plt.ylabel("Evaluation Metrics")
    plt.title("Searching for the best K value of K-Nearest Neighbors")
    plt.legend()
    plt.show()

# ============================================

if __name__ == '__main__':
    start_time = time.time()

    train_data = None
    test_data = None
    if MODE == TransformMode.STANDARDIZE_SELECTED_COLUMNS:
        train_data = pd.read_csv(STANDARDIZED_TRAIN_DATA_PATH)
        test_data = pd.read_csv(STANDARDIZED_TEST_DATA_PATH)
    elif MODE == TransformMode.SCALE_WITH_RANGE or MODE == TransformMode.SCALE_SELECTED_COLUMNS:
        train_data = pd.read_csv(SCALED_TRAIN_DATA_PATH)
        test_data = pd.read_csv(SCALED_TEST_DATA_PATH)
    else:
        #train_data = pd.read_csv(TRAIN_DATA_PATH)
        #test_data = pd.read_csv(TEST_DATA_PATH)
        train_data = pd.read_csv(ONE_HOT_TRAIN_PATH_70)
        test_data = pd.read_csv(ONE_HOT_TEST_PATH_70)

    #knn(train_data, test_data)
    knn_search_K(train_data, test_data)

    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))