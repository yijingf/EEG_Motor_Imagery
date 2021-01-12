# This code defines functions for three ML methods in order to assess baseline performace
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# to use commonly default settings, call SVM(), RandForest(), LogReg()
def SVM(train_data, train_labels, test_data, test_labels, verbose = 1):
    clf = SVC(C = 1, gamma = 'scale')
    clf.fit(train_data, train_labels)
    pred_labels = clf.predict(test_data)
    if verbose != -1:
        print('\n Test Accuracy: %f' % accuracy_score(test_labels, pred_labels))
    return(pred_labels, accuracy_score(test_labels, pred_labels))

def RandForest(train_data, train_labels, test_data, test_labels, verbose = 1):
    clf = RandomForestClassifier(n_estimators = 100)
    clf.fit(train_data, train_labels)
    pred_labels = clf.predict(test_data)
    if verbose != -1:
        print('\n Test Accuracy: %f' % accuracy_score(test_labels, pred_labels))
    return(pred_labels, accuracy_score(test_labels, pred_labels))


# All Baselines using default parameters
def CV_Baselines(data, labels, test_data, test_labels, K = 0, verbose = 1):
    
    clfs = [
        SVC(C = 1, gamma = 'scale'),
        LogisticRegression(penalty = 'l1', C = 1, solver='liblinear'),
        RandomForestClassifier(n_estimators = 100)
        ]
    
    if K > 0:
        acc_dict = dict()
        for clf in clfs:
            acc_dict[str(clf.__class__.__name__)] = cross_val_score(clf, data, labels, cv = K).mean()
            if verbose != -1:
                print(f'classifier: ', clf.__class__.__name__, 'avg CV acc: ', acc_dict[str(clf.__class__.__name__)])
    
    if K == 0:
        acc_dict = dict()
        for clf in clfs:
            clf.fit(data, labels)
            pred_labels = clf.predict(test_data)
            acc_dict[str(clf.__class__.__name__)] = accuracy_score(test_labels, pred_labels)
            if verbose != -1:
                print(f'classifier: ', clf.__class__.__name__, 'test acc: ', acc_dict[str(clf.__class__.__name__)])
    
    return(acc_dict)


# to use hyperparameters chosen by grid search, call SVM_gridSearch(), LogReg_gridSearch(), RandForest_gridSearch()
def SVM_gridSearch(data, labels, test_data, test_labels, K, verbose=1):
    """    
    Support Vector Machine
    input: data and labels should contain train and validation sets
    C is selected by grid-search in the range of {0.01, 0.1, 1, 10, 100}.
    """
    def SVM_CV(data, labels, K):
        #perform cross-validation and SVM classification
        hyperparams = {'C': [0.01, 0.1, 1, 10, 100],
                       'gamma': np.arange(0.1, 10, 0.1)}
        cross_valid = StratifiedKFold(n_splits = K, 
                                      shuffle=True, random_state = 0)
        grid_search = GridSearchCV(SVC(), 
                                   param_grid = hyperparams, 
                                   cv = cross_valid)
        grid_search.fit(data, labels)
        return grid_search.best_params_, grid_search.best_score_
    
    PARAM, score = SVM_CV(data=data, labels=labels, K = K)
    
    if verbose != -1:
        print('f best parameter: ', PARAM)
        print(f'best CV acc', score) 
    
    clf = SVC(C = PARAM['C'], gamma = PARAM['gamma'])
    clf.fit(data, labels)
    pred_labels = clf.predict(test_data)
    
    if verbose != -1:
        print('test acc: %f' % accuracy_score(test_labels, pred_labels))
    
    return(pred_labels, accuracy_score(test_labels, pred_labels))


def RandForest_gridSearch(data, labels, test_data, test_labels, K, verbose=1):
    """
    Random Forest 
    The number of estimators is chosen in {5, 10, 20, 50, 100, 500, 1000}.
    """
    def RandForest_CV(data, labels, K):
        # perform cross-validation for RF classification
        hyperparams = {'n_estimators' : [5, 10, 20, 50, 100, 500, 1000],
                       'min_samples_split': [2, 5, 10]}
        cross_valid = StratifiedKFold(n_splits = K, 
                                      shuffle = True, random_state = 0)
        grid_search = GridSearchCV(RandomForestClassifier(), 
                                   param_grid = hyperparams, 
                                   cv = cross_valid)
        grid_search.fit(data, labels)
        return grid_search.best_params_, grid_search.best_score_
    
    PARAM, score = RandForest_CV(data=data, labels=labels, K=K)
    
    if verbose != -1:
        print('f best parameter: ', PARAM)
        print(f'best CV acc', score) 
    
    clf = RandomForestClassifier(n_estimators = PARAM['n_estimators'], 
                                 min_samples_split = PARAM['min_samples_split'])
    clf.fit(data, labels)
    pred_labels = clf.predict(test_data)
    
    if verbose != -1:
        print('test acc: %f' % accuracy_score(test_labels, pred_labels))
    
    return(pred_labels, accuracy_score(test_labels, pred_labels))



# Generate data
from sklearn.datasets import make_blobs
cluster_std = 5
num_features_for_samples = 4
num_samples_total = 1000
test_ratio = 0.20
X, y = make_blobs(n_samples = num_samples_total, 
                  n_features = num_features_for_samples, 
                  cluster_std = cluster_std)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_ratio)

# test function
acc_dict = CV_Baselines(X_train, y_train, X_test, y_test, K = 3)
RandForest_gridSearch(data = X_train, labels = y_train, 
                      test_data = X_test, test_labels = y_test, K = 3)