import enum
import time
# import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

TOY_DATA_PATH = "../toy.csv"
DATA_PATH = "../data/ks-projects-201801.csv"
PROCESSED_TOY_DATA_PATH = "../toy_processed.csv"

PROCESSED_TRAIN_PATH = "../data/train.csv"
PROCESSED_TEST_PATH = "../data/test.csv"

PROCESSED80_TRAIN_PATH = "../data/train_80.csv"
PROCESSED80_TEST_PATH = "../data/test_80.csv"

PROCESSED95_TRAIN_PATH = "../data/train_95.csv"
PROCESSED95_TEST_PATH = "../data/test_95.csv"

PROCESSED99_TRAIN_PATH = "../data/train_99.csv"
PROCESSED99_TEST_PATH = "../data/test_99.csv"

STANDARDIZED_TRAIN_PATH = "../data/train_standardized.csv"
STANDARDIZED_TEST_PATH = "../data/test_standardized.csv"

SCALED_TRAIN_PATH = "../data/train_scaled.csv"
SCALED_TEST_PATH = "../data/test_scaled.csv"

ONE_HOT_TRAIN_PATH_70 = "../data/train_one_hot_70.csv"
ONE_HOT_TEST_PATH_70 = "../data/test_one_hot_70.csv"

ONE_HOT_TRAIN_PATH_80 = "../data/train_one_hot_80.csv"
ONE_HOT_TEST_PATH_80 = "../data/test_one_hot_80.csv"

FEATURES = ['main_category', 'backers', 'country', 'usd_goal_real', 'duration_in_days']
FEATURES_TO_PLOT = ["main_category", "country", "state"]
FEATURES_TO_DROP = ['ID', 'name', 'category', 'currency', 'goal', 'pledged', 'usd pledged', 'usd_pledged_real']
FEATURES_TO_ENCODE = ["main_category", "country"]
FEATURES_TO_STANDARDIZE = ['backers', 'usd_goal_real', 'duration_in_days']

class TransformMode(enum.Enum):
   NONE = 0
    # Whether to standardize columns in FEATURES_TO_STANDARDIZE
   STANDARDIZE_SELECTED_COLUMNS = 1
    # Whether to scale all FEATURES, final range (0,20)
   SCALE_WITH_RANGE = 2
    # Only scale 'backers', 'usd_goal_real', 'duration_in_days', using default range
   SCALE_SELECTED_COLUMNS = 3

MODE = TransformMode.NONE

'''
    Steps:
    1) Drop unnecessary features
    2) Drop rows whose 'state' is not in ("successful", "failed")
    3) Create a new feature 'duration_in_days' based on the difference between deadline and launched date
    4) One-hot encode categorical data ('main_category' and 'country'), convert 'state' column from String to Integer (1 or 0)
    5) (Skipped) Standardize numerical columns ('backers', 'usd_goal_real', 'duration_in_days')
    6) Randomize/shuffle，and split into train/test set with a ratio of 9:1 (or 7:3, 8:2)
'''
def pre_process():
    df = pd.read_csv(DATA_PATH)
    #examine_data(df)
    #plot_with_pie_chart(df)
    #plot_target_categories(df)

    # Drop the features that are not needed
    drop_features(df)
    # print(" --- number of rows before = {}".format(len(df)))
    df = drop_rows(df)
    # print(" --- number of rows after = {}".format(len(df)))
    df = encode_columns(df)

    # Create a new column "duration_in_days" by calculating
    # the number of days difference between "deadline" and "launched"
    convert_duration_fast(df)

    if MODE == TransformMode.STANDARDIZE_SELECTED_COLUMNS:
        standardize(df)
    elif MODE == TransformMode.SCALE_WITH_RANGE:
        scale(df)
    elif MODE == TransformMode.SCALE_SELECTED_COLUMNS:
        scale_selected_features(df)

    # split data into 70% training set, 30% test set
    train_set, test_set = shuffle_and_split(df, 0.3)
    examine_data(df)

    # df.to_csv(PROCESSED_TOY_DATA_PATH, index=False)

    if MODE == TransformMode.STANDARDIZE_SELECTED_COLUMNS:
        train_set.to_csv(STANDARDIZED_TRAIN_PATH, index=False)
        test_set.to_csv(STANDARDIZED_TEST_PATH, index=False)
    elif MODE == TransformMode.SCALE_WITH_RANGE or MODE == TransformMode.SCALE_SELECTED_COLUMNS:
        train_set.to_csv(SCALED_TRAIN_PATH, index=False)
        test_set.to_csv(SCALED_TEST_PATH, index=False)
    else:
        #train_set.to_csv(PROCESSED_TRAIN_PATH, index=False)
        #test_set.to_csv(PROCESSED_TEST_PATH, index=False)
        train_set.to_csv(ONE_HOT_TRAIN_PATH_70, index=False)
        test_set.to_csv(ONE_HOT_TEST_PATH_70, index=False)

# Plot the y column using a Seaborn countplot
def plot_target_categories(df):
    sns.countplot(x="state", data=df, palette='hls')
    plt.xlabel("Funding Status")
    plt.ylabel("Count")
    plt.title("Count of Funding Outcome")
    plt.show()

def plot_with_pie_chart(df):
    '''
    Categorical data has a categories and a ordered property, which list their possible values
    and whether the ordering matters or not. These properties are exposed as s.cat.categories and s.cat.ordered.
    If you don’t manually specify categories and ordering, they are inferred from the passed arguments.
    '''
    for feature in FEATURES_TO_PLOT:
        labels = df[feature].astype('category').cat.categories.tolist()
        counts = df[feature].value_counts()
        sizes = [counts[var_cat] for var_cat in labels]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)  # autopct is show the % on plot
        ax1.axis('equal')
        #plt.title("Feature Distribution - " + str(feature))
        ax1.set_title("Feature Distribution - " + str(feature), pad=20)
        plt.show()

def examine_data(df):
    print("\n ------ Examining data... ------")
    print(df.shape)
    #print(df.info())
    #print("List of attributes: ", list(df.columns))
    #print("Total number of attributes: ", len(list(df.columns))-1)

    '''
    backers: min=0, max=219382
    usd_goal_real: min=0.01, max=166361390.71
    duration_in_days: min=0, max=91
    
    for feature in FEATURES_TO_STANDARDIZE:
        print("{}: min={}, max={}".format(feature, df[feature].min(), df[feature].max()))
    '''

    '''
    # Print number of null values (3801 in total)
    print("Number of null values={}".format(df.isnull().values.sum()))
    
    # Column-wise distribution of null values
    # The result shows there are 4 null values in 'name' column, 
    # and 3797 in 'usd pledged' column
    # no need to worry since I don't use these 2 columns anyway
    print(df.isnull().sum())
    '''

    '''
    # print a table of main_category : count
    # Total categories = 15
    print(df['main_category'].value_counts())
    # two ways to get the number of unique values in main_category
    print(df['main_category'].value_counts().count())
    print("Number of unique project categories = {}".format(df['main_category'].nunique()))
    '''

    '''
    # Total defined countries: 22.
    # The only undefined countries "N,0" will occur with state = "undefined", and it's dropped already
    print(df['country'].value_counts())
    print(df['country'].value_counts().count())
    '''

    print("backers: mean={}, median={}, std={}".format(df['backers'].mean(), df['backers'].median(), df['backers'].std()))
    print("usd goal: mean={}, median={}, std={}".format(df['usd_goal_real'].mean(), df['usd_goal_real'].median(), df['usd_goal_real'].std()))
    print("duration in days: mean={}, median={}, std={}".format(df['duration_in_days'].mean(), df['duration_in_days'].median(), df['duration_in_days'].std()))
    print("------ Examination Done! ------")

def drop_features(df):
    for feature in FEATURES_TO_DROP:
        df.drop([feature], axis=1, inplace=True)

# drop all rows whose |state| is not in ("successful", "failed")
def drop_rows(df):
    # df['state']:  (0, 'successful') (1, 'failed') (2, 'failed') (3, 'undefined') (4, 'canceled') ...
    successRows = df[df['state'] == 'successful']
    failedRows = df[df['state'] == 'failed']
    tmp_df = pd.concat([successRows, failedRows])

    # Filter out rows whose 'country' has an invalid value
    return tmp_df[~tmp_df['country'].str.contains("N,0")]
    #validCountries = df[~df['country'].str.contains("N,0")]
    #return pd.concat([successRows, failedRows])

def encode_columns(df):
    # There are 15 categories in total
    # 22 countries in total (excluding the one country that's invalid - N,0)
    '''
    encoder = LabelEncoder()
    for feature in FEATURES_TO_ENCODE:
        df[feature] = encoder.fit_transform(df[feature])

    '''
    # one-hot encode 'country' column
    country_one_hot = pd.get_dummies(df.country, prefix='is_country_')
    df.drop(['country'], axis=1, inplace=True)
    df = pd.concat([df, country_one_hot], axis=1)

    # one-hot encode 'main_category' column
    category_one_hot = pd.get_dummies(df.main_category, prefix='is_category_')
    df.drop(['main_category'], axis=1, inplace=True)
    df = pd.concat([df, category_one_hot], axis=1)

    # Convert state: 'successful' to 1, 'failed' to 0
    df['state'] = df['state'].astype(str)
    df['state'] = np.where(df["state"].str.contains("successful"), 1, 0)
    print("Encoding done!!!")
    print(df.head())
    return df

'''
# The original slow version. DO NOT USE!!!
def convert_duration(df):
    days = []
    for index, row in df.iterrows():
        startDate = str(row['launched']).split()[0]
        endDate = str(row['deadline'])
        start = datetime.datetime.strptime(startDate, "%Y-%m-%d").date()
        end = datetime.datetime.strptime(endDate, "%Y-%m-%d").date()
        duration = (end-start).days
        days.append(duration)
        # print("startDate={}, endDate={}, days={}".format(startDate, endDate, duration))
        print("idx={}, deadline={}, launched={}".format(index, row['deadline'], row['launched']))
    # add the new feature
    df['duration_in_days'] = days

    df.drop(['launched'], axis=1, inplace=True)
    df.drop(['deadline'], axis=1, inplace=True)
'''

def convert_duration_fast(df):
    df['launched'] = pd.to_datetime(df['launched'])
    df['deadline'] = pd.to_datetime(df['deadline'])
    df['duration_in_days'] = (df['deadline'] - df['launched']).dt.days

    df.drop(['launched'], axis=1, inplace=True)
    df.drop(['deadline'], axis=1, inplace=True)

def standardize(df):
    # df[FEATURES_TO_TRANSFORM] = df[FEATURES_TO_TRANSFORM].apply(lambda x: StandardScaler().fit_transform(x))
    features = df[FEATURES_TO_STANDARDIZE]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    df[FEATURES_TO_STANDARDIZE] = features

def scale(df):
    features = df[FEATURES]
    scaler = MinMaxScaler(feature_range=(0, 20)).fit(features.values)
    features = scaler.transform(features.values)

    df[FEATURES] = features

def scale_selected_features(df):
    features = df[['backers', 'usd_goal_real', 'duration_in_days']]
    scaler = MinMaxScaler().fit(features.values)
    features = scaler.transform(features.values)

    df[['backers', 'usd_goal_real', 'duration_in_days']] = features

# Adapted from page 52 of the book
# Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd Edition)
# Total rows: 331675
def shuffle_and_split(data, test_ratio):
    size = len(data)
    shuffled_indices = np.random.permutation(size)
    test_set_size = int(size * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == '__main__':
    start_time = time.time()
    pre_process()
    end_time = time.time()
    print("Time taken: {}".format(end_time - start_time))