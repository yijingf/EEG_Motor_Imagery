import scipy.io
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def gridSearch_baseline_models(data, labels, test_ratio = 0.2, K = 0, verbose = 1):

    min_max_scaler = preprocessing.MinMaxScaler()
    data_minmax = min_max_scaler.fit_transform(data)

    X_train, X_test, y_train, y_test = train_test_split(data_minmax, labels, test_size = test_ratio)

    if K == 0:

        clfs = [
            SVC(C = 1, gamma = 'scale'),
            LogisticRegression(penalty = 'l1', C = 1, solver='liblinear'),
            RandomForestClassifier(n_estimators = 100)
        ]

        acc_dict = dict()

        for clf in clfs:

            clf.fit(X_train, y_train)
            pred_labels = clf.predict(X_test)

            acc_dict[str(clf.__class__.__name__)] = accuracy_score(y_test, pred_labels)

            if verbose != -1:
                print(f'clf: ', clf.__class__.__name__, '\n test acc: ', acc_dict[str(clf.__class__.__name__)])


    if K > 0:

        clfs = [
            SVC(),
            LogisticRegression(),
            RandomForestClassifier()
        ]

        acc_dict = dict()

        for clf in clfs:

            if clf.__class__.__name__ == 'SVC':
                hyperparams = {'C': [0.01, 0.1, 1, 10, 100],
                               'gamma': np.arange(0.1, 10, 0.1)}

            if clf.__class__.__name__ == 'LogisticRegression':
                hyperparams = {'C': [0.01, 0.1, 1, 10, 100, 1000]}

            if clf.__class__.__name__ == 'RandomForestClassifier':
                hyperparams = {'n_estimators' : [5, 10, 20, 50, 100, 500, 1000],
                               'min_samples_split': [2, 5, 10]}

            cross_valid = StratifiedKFold(n_splits = K,
                                          shuffle = True, random_state = 0)
            grid_search = GridSearchCV(clf,
                                       param_grid = hyperparams,
                                       cv = cross_valid)
            grid_search.fit(X_train, y_train)

            PARAM, score = grid_search.best_params_, grid_search.best_score_

            if verbose != -1:
                print(f'clf: ', clf.__class__.__name__,
                      '\n best parameter: ', PARAM,
                      '\n best CV acc: ', score)

            if clf.__class__.__name__ == 'SVC':
                model = SVC(C = PARAM['C'], gamma = PARAM['gamma'])

            if clf.__class__.__name__ == 'LogisticRegression':
                model = LogisticRegression(C = PARAM['C'])

            if clf.__class__.__name__ == 'RandomForestClassifier':
                model = RandomForestClassifier(n_estimators = PARAM['n_estimators'],
                                               min_samples_split = PARAM['min_samples_split'])

            model.fit(X_train, y_train)
            pred_labels = model.predict(X_test)

            acc_dict[str(clf.__class__.__name__)] = accuracy_score(y_test, pred_labels)

            if verbose != -1:
                print(f' test acc: ', acc_dict[str(clf.__class__.__name__)])

    return(acc_dict)