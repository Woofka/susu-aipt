import numpy as np
import pandas as pd
import graphviz
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


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


def graph_desicion_tree(df, filename):
    X, y = split_data(df)
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(X, y)

    dot_data = tree.export_graphviz(clf, out_file=None, max_depth=4, feature_names=list(X.columns))
    graph = graphviz.Source(dot_data)
    graph.render(cleanup=True, outfile=filename, format='png')


def classify(df, test_size, criterion):
    X_train, X_test, y_train, y_test = split_data(df, test_size)
    clf = DecisionTreeClassifier(criterion=criterion)
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


def run_compare(df, title, filename):
    test_sizes = [0.4, 0.3, 0.2, 0.1]
    criterions = ['gini', 'entropy']
    avg_of_n = 100

    results = {}

    for criterion in criterions:
        results[criterion] = {
            'x': test_sizes,
            'acc': [],
            'rs': [],
            'ps': [],
            'f1': []
        }
        for test_size in test_sizes:
            print(f'Running criterion {criterion} test_size {test_size}')
            acc = []
            rs = []
            ps = []
            f1 = []
            for i in range(avg_of_n):
                _, metrics = classify(df, test_size, criterion)
                acc.append(metrics['acc'])
                rs.append(metrics['rs'])
                ps.append(metrics['ps'])
                f1.append(metrics['f1'])
            results[criterion]['acc'].append(sum(acc)/len(acc))
            results[criterion]['rs'].append(sum(rs)/len(rs))
            results[criterion]['ps'].append(sum(ps)/len(ps))
            results[criterion]['f1'].append(sum(f1)/len(f1))

    fig, axs = plt.subplots(len(criterions), 1)
    for idx, (criterion, data) in enumerate(results.items()):
        axs[idx].plot(data['x'], data['acc'], label='Accuracy')
        axs[idx].plot(data['x'], data['rs'], label='Recall')
        axs[idx].plot(data['x'], data['ps'], label='Precision')
        axs[idx].plot(data['x'], data['f1'], label='F1')

        axs[idx].set_title(criterion, loc='right')
        axs[idx].set_xlabel('Test size')
        axs[idx].set_ylabel('Metric value')
        axs[idx].grid(True)
        axs[idx].legend()

    axs[0].set_title(title)
    fig.tight_layout()
    plt.savefig(filename)


def main():
    df_adult = load_df_adult()
    df_grades = load_df_grades()

    print('df_adult shape:', df_adult.shape)
    print(df_adult['salary'].value_counts())
    print('df_grades shape:', df_grades.shape)
    print(df_grades['GRADE'].value_counts())

    graph_desicion_tree(df_adult, 'graph_adult.png')
    graph_desicion_tree(df_grades, 'graph_grades.png')

    run_compare(df_grades, 'Dependence of metrics on test size (grades dataset)', 'plot_grades.png')
    run_compare(df_adult, 'Dependence of metrics on test size (adult dataset)', 'plot_adult.png')


if __name__ == '__main__':
    main()
