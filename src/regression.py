import enum
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.utils.fixes import loguniform


TRAIN_DATA_PATH = "../data/train.csv"
TEST_DATA_PATH = "../data/test.csv"

TRAIN80_DATA_PATH = "../data/train_80.csv"
TEST80_DATA_PATH = "../data/test_80.csv"

TRAIN95_DATA_PATH = "../data/train_95.csv"
TEST95_DATA_PATH = "../data/test_95.csv"

TRAIN99_DATA_PATH = "../data/train_99.csv"
TEST99_DATA_PATH = "../data/test_99.csv"

ONE_HOT_TRAIN_PATH_70 = "../data/train_one_hot_70.csv"
ONE_HOT_TEST_PATH_70 = "../data/test_one_hot_70.csv"

STANDARDIZED_TRAIN_DATA_PATH = "../train_standardized.csv"
STANDARDIZED_TEST_DATA_PATH = "../test_standardized.csv"

SCALED_TRAIN_DATA_PATH = "../data/train_scaled.csv"
SCALED_TEST_DATA_PATH = "../data/test_scaled.csv"

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


def logistic_regression(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    # If the only features to train on are "duration_in_days"
    # train score = 0.6, test score = 0.6
    # train_X = train_data.iloc[:, [5]]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]
    # test_X = test_data.iloc[:, [5]]
    # examine_data(test_data)

    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(train_X, train_y)
    pred_y = LR.predict(test_X)

    evaluate(LR, test_X, test_y, pred_y)

    '''
    # Use score to evaluate
    train_score = LR.score(train_X, train_y)
    test_score = LR.score(test_X, test_y)
    print("train score: {}".format(train_score))
    print("test score: {}".format(test_score))
    '''

def sgd(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]

    test_y = test_data['state']
    test_X = test_data.iloc[:, FEATURES_INDICES]

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # loss="log" - logsitic regression
    # shuffle=False, since the data has already been shuffled during pre-processing
    SGD = SGDClassifier(loss="log", penalty="l2", early_stopping=True, max_iter=100, shuffle=False, verbose=1)
    # SGD.partial_fit(train_X, train_y, classes=np.unique(train_y))
    SGD.fit(train_X, train_y)
    # new_weights = SGD.coef_
    #print("Learned weights: {}".format(SGD.coef_))
    #print("Converged after {} iterations".format(SGD.n_iter_))

    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    plot(loss_history)

    pred_y = SGD.predict(test_X)
    evaluate(SGD, test_X, test_y, pred_y)

    cmap = plt.get_cmap('Blues')
    plot_confusion_matrix(SGD, test_X, test_y, cmap=cmap)
    plt.show()

def plot(loss_history):
    loss_list = []
    for line in loss_history.split('\n'):
        if (len(line.split("loss: ")) == 1):
            continue
        loss_list.append(float(line.split("loss: ")[-1]))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Logistic Regression with Stochastic Gradient Descent")
    plt.show()

'''
def grid_search(train_data, test_data):
    train_y = train_data['state']
    train_X = train_data.iloc[:, FEATURES_INDICES]
    #param_grid = {'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
    param_dist = {'alpha': loguniform(1e-4, 1e0)}

    SGD = SGDClassifier(loss="log", penalty="l2", max_iter=350)
    random_search = RandomizedSearchCV(SGD, param_distributions=param_dist,
                                       n_iter=50)
    random_search.fit(train_X, train_y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), 50))
    report(random_search.cv_results_)
'''

# use mode to predict (always predict "failed")
def baseline_mode(train_data, test_data):
    # training data: 0 - 177863, 1 - 120645
    # test data: 0 - 19856, 1 - 13311

    # mode_freq = train_data['state'].value_counts().max()
    # print(train_data['state'].value_counts())

    # Get the mode (0, since mode is "failed")
    mode = train_data['state'].value_counts().idxmax()

    # How many times does 0 occur in the state column in test set?
    count_in_test = (test_data.state == mode).sum()
    # Total number of rows in test set
    test_size = len(test_data.index)
    accuracy = float(count_in_test / test_size)
    f1_score = float(count_in_test/(count_in_test + 0.5*(test_size - count_in_test)))
    print("accuracy={}, f1_score={}".format(accuracy, f1_score))

def baseline_single_feature(train_data, test_data):
    train_y = train_data['state']

    best_feature = None
    best_score = 0
    best_confusion_matrix = None
    best_pred = None

    test_y = test_data['state']
    for feature, feature_idx in zip(FEATURES, FEATURES_INDICES):
        train_X = train_data.iloc[:, [feature_idx]]
        test_X = test_data.iloc[:, [feature_idx]]
        LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(train_X, train_y)
        pred_y = LR.predict(test_X)
        score, confusion = evaluate(LR, test_X, test_y, pred_y, False)
        if score > best_score:
            best_score = score
            best_feature = feature
            best_confusion_matrix = confusion
            best_pred = pred_y
    print_metrics(best_score, best_confusion_matrix, test_y, best_pred, best_feature)

def examine_data(df):
    print("\n ------ Examining data... ------")
    # print(df.shape)
    # print(df.head)
    print(df['state'].value_counts())
    print("------ Examination Done! ------")

def evaluate(model, test_X, test_y, pred_y, do_print=True):
    '''
    lbfgs: (18818+11094)/33167 = 0.90186
    liblinear: (18860+11021)/33167 = 0.9009
    newton-cg: (18884+10963)/33167 = 0.8999
    '''
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

    #logistic_regression(train_data, test_data)
    #baseline_mode(train_data, test_data)
    #baseline_single_feature(train_data, test_data)
    sgd(train_data, test_data)
    #grid_search(train_data, test_data)

    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))