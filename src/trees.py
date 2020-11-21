import enum
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

ONE_HOT_TRAIN_PATH_70 = "../data/train_one_hot_70.csv"
ONE_HOT_TEST_PATH_70 = "../data/test_one_hot_70.csv"

FEATURES = ['backers', 'usd_goal_real', 'is_country__AT','is_country__AU','is_country__BE','is_country__CA','is_country__CH','is_country__DE','is_country__DK','is_country__ES','is_country__FR','is_country__GB','is_country__HK','is_country__IE','is_country__IT','is_country__JP','is_country__LU','is_country__MX','is_country__NL','is_country__NO','is_country__NZ','is_country__SE','is_country__SG','is_country__US','is_category__Art','is_category__Comics','is_category__Crafts','is_category__Dance','is_category__Design','is_category__Fashion','is_category__Film & Video','is_category__Food','is_category__Games','is_category__Journalism','is_category__Music','is_category__Photography','is_category__Publishing','is_category__Technology','is_category__Theater', 'duration_in_days']
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

def decision_tree(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    DT = DecisionTreeClassifier(criterion="entropy", max_depth=11)

    #DT = do_search_trees(DT, train_X, train_y)
    DT.fit(train_X, train_y)

    #tree.plot_tree(DT)
    #plt.show()
    pred_y = DT.predict(test_X)
    plot_tree(DT, 8)
    evaluate(DT, test_X, test_y, pred_y)

def random_forest(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    #RF = do_search_forest(train_X, train_y)
    RF = RandomForestClassifier(n_estimators=70, max_depth=20, criterion='entropy')
    RF.fit(train_X, train_y)
    pred_y = RF.predict(test_X)
    plot_forest(RF, 8)
    evaluate(RF, test_X, test_y, pred_y)

def do_search_both(train_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]
    do_search_trees(train_X, train_y)
    do_search_forest(train_X, train_y)

def do_search_trees(train_X, train_y):
    param_max_depth = list(range(8, 12))
    param_criterion = ['gini', 'entropy']
    param_grid = [{'max_depth': param_max_depth
                   #'criterion': param_criterion
                   }]
    gs = GridSearchCV(estimator=DecisionTreeClassifier(criterion='entropy'),
                      param_grid=param_grid,
                      cv=5,
                      n_jobs=-1,
                      scoring='accuracy')

    gs.fit(train_X, train_y)
    print("Best score from grid search (DT): {}".format(gs.best_score_))
    print("Best parameters from grid search (DT): {}".format(gs.best_params_))
    return gs.best_estimator_

def do_search_forest(train_X, train_y):
    param_max_depth = [10, 15, 20]
    param_criterion = ['gini', 'entropy']
    param_n_estimators = [10, 20, 70]
    param_grid = [{'max_depth': param_max_depth,
                   #'criterion': param_criterion,
                   'n_estimators': param_n_estimators}]
    gs = GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                      param_grid=param_grid,
                      n_jobs=-1,
                      scoring='accuracy')
    gs.fit(train_X, train_y)
    print("Best score from grid search (RF): {}".format(gs.best_score_))
    print("Best parameters from grid search (RF): {}".format(gs.best_params_))
    return gs.best_estimator_

def evaluate(model, test_X, test_y, pred_y, do_print=True):
    confusion = confusion_matrix(test_y, pred_y)
    score = model.score(test_X, test_y)
    accuracy = accuracy_score(test_y, pred_y)
    result = precision_recall_fscore_support(test_y, pred_y)
    f1 = result[2][0]
    if do_print:
        print_metrics(model, score, confusion, test_y, pred_y)
    return accuracy, f1

def print_metrics(tree_model, score, confusion_matrix, test_y, pred_y, feature=None):
    if feature:
        print("Best feature: {}".format(feature))
    print(confusion_matrix)
    print("test score: {}".format(score))
    print(classification_report(test_y, pred_y, target_names=['0', '1']))
    #tree.plot_tree(tree_model)
    #plt.show()

def plot_tree(DT, n):
    feat_importances = pd.Series(DT.feature_importances_, index=FEATURES)
    feat_importances.nlargest(n).plot(kind='barh')
    plt.title("Feature Importance (Decision Trees)")
    plt.show()

def plot_forest(RF, n):
    feat_importances = pd.Series(RF.feature_importances_, index=FEATURES)
    feat_importances.nlargest(n).plot(kind='barh')
    plt.title("Feature Importance (Random Forest)")
    plt.show()

# ============================================

if __name__ == '__main__':
    start_time = time.time()

    # train_data = pd.read_csv(TRAIN_DATA_PATH)
    # test_data = pd.read_csv(TEST_DATA_PATH)
    train_data = pd.read_csv(ONE_HOT_TRAIN_PATH_70)
    test_data = pd.read_csv(ONE_HOT_TEST_PATH_70)

    #decision_tree(train_data, test_data)
    random_forest(train_data, test_data)

    #do_search_both(train_data)

    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))