import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt


lab3_metrics_grades = {
    'acc': 0.4593333333333329,
    'rs': 0.4593333333333329,
    'ps': 0.6844000000000002,
    'f1': 0.4847523809523814
}

lab3_metrics_adult = {
    'acc': 0.8121894672194074,
    'rs': 0.8121894672194074,
    'ps': 0.8151834167196764,
    'f1': 0.8135772455607967
}


def load_df_adult():
    df = pd.read_csv(
        'adult.csv',
        names=['age', 'workclas', 'State-gov', 'fnlwgt', 'education', 'marital-status', 'occupation', 'relationship',
               'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
    )
    df = df.dropna()

    encoder = LabelEncoder()
    features_to_encode = ['fnlwgt', 'workclas', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                          'native-country']
    for feature in features_to_encode:
        df[feature] = encoder.fit_transform(df[feature])

    return df


def load_df_grades():
    df = pd.read_csv('grades.csv')
    df = df.dropna()

    encoder = LabelEncoder()
    features_to_encode = ['PUPIL_SEX', 'PUPIL_CLASS']
    for feature in features_to_encode:
        df[feature] = encoder.fit_transform(df[feature])

    return df


def split_data(data, test_size=None):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if test_size is None:
        return X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test


def classify(df, n_estimators):
    X_train, X_test, y_train, y_test = split_data(df, 0.2)
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=0.01, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_test = np.array(y_test)

    metrics = {
        'acc': accuracy_score(y_test, y_pred),
        'rs': recall_score(y_test, y_pred, average="weighted", zero_division=0),
        'ps': precision_score(y_test, y_pred, average="weighted", zero_division=0),
        'f1': f1_score(y_test, y_pred, average="weighted")
    }

    return clf, metrics


def run_compare(df, title, filename, tree_res=None):
    n_estimators_list = list(range(50, 101, 10))
    avg_of_n = 10

    results = {
        'acc': [],
        'rs': [],
        'ps': [],
        'f1': []
    }
    for n_estimators in n_estimators_list:
        print(f'Running {n_estimators} estimators')
        acc = []
        rs = []
        ps = []
        f1 = []
        for i in range(avg_of_n):
            _, metrics = classify(df, n_estimators)
            acc.append(metrics['acc'])
            rs.append(metrics['rs'])
            ps.append(metrics['ps'])
            f1.append(metrics['f1'])
        results['acc'].append(sum(acc)/len(acc))
        results['rs'].append(sum(rs)/len(rs))
        results['ps'].append(sum(ps)/len(ps))
        results['f1'].append(sum(f1)/len(f1))

    fig, ax = plt.subplots()

    ax.plot(n_estimators_list, results['acc'], 'b', label='Accuracy')
    ax.plot(n_estimators_list, results['rs'], 'k', label='Recall')
    ax.plot(n_estimators_list, results['ps'], 'g', label='Precision')
    ax.plot(n_estimators_list, results['f1'], 'r', label='F1')

    if tree_res is not None:
        ax.plot(n_estimators_list, [tree_res['acc']] * len(n_estimators_list), 'b--')
        ax.plot(n_estimators_list, [tree_res['rs']] * len(n_estimators_list), 'k--')
        ax.plot(n_estimators_list, [tree_res['ps']] * len(n_estimators_list), 'g--')
        ax.plot(n_estimators_list, [tree_res['f1']] * len(n_estimators_list), 'r--')

    ax.set_xlabel('Number of members')
    ax.set_ylabel('Metric value')
    ax.grid(True)
    ax.legend()

    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(filename)


def main():
    df_adult = load_df_adult()
    df_grades = load_df_grades()

    print('df_adult shape:', df_adult.shape)
    print(df_adult['salary'].value_counts())
    print('df_grades shape:', df_grades.shape)
    print(df_grades['GRADE'].value_counts())

    run_compare(df_grades, 'Dependence of metrics on number of members (grades dataset)', 'plot_grades_2.png',
                lab3_metrics_grades)
    run_compare(df_adult, 'Dependence of metrics on number of members (adult dataset)', 'plot_adult_2.png',
                lab3_metrics_adult)


if __name__ == '__main__':
    main()
