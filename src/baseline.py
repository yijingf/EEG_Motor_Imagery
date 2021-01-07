# import numpy as np
# from sklearn import svm
# from sklearn.metrics import accuracy_score
# from eeg_loader import *
# from config import SUBs, freq_bands
# from bandpower import *

# # Preprocess config
# l_freq = 4
# h_freq = 30
# resample_freq = 100 # original sfreq is 160

# # load data
# data_loader = DataLoader(window_len=100, overlap=0) # without overlapping, window_len=100 unit?
# # take a window sufficiently long to encompasses at least two full cycles of the lowest frequency of interest. 
# # https://raphaelvallat.com/bandpower.html
# # 4hz -> 0.25s -> 2 cycles 0.5s -> resample_freq=100: 50 samples? 

# X, y, _ = data_loader.load_data(SUBs, 
#                                 l_freq=l_freq, h_freq=h_freq, 
#                                 resample_sfreq=resample_freq, mesh=False)
# X = np.array([getBandPower_Pool(x, freq_bands, resample_freq) for x in X])

# # X is a matrix with the shape of (n, 3, 64) where n is the number of samples?, 3 is the number of frequency bands and 64 is the number of channels

# # to do: train/valid/test split 60 20 20 




import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def SVM(data, labels, K, verbose=1):
    '''
    Support Vector Machine
    
    C and γ were selected by grid-search in the range 
    of {0.01, 0.1, 1, 10, 100} and {0.1, 0.2, ..., 1, 2, ..., 10} respectively.
    
    '''
    def CV_SVM(data, labels, K):
        #perform cross-validation and SVM classification
        hyperparams = {'C' : [0.01, 0.1, 1, 10, 100],
                       'gamma': np.arange(0.1, 10, 0.1) }
        cross_valid = StratifiedKFold(n_splits = K, 
                                      shuffle=True, random_state = 0)
        grid_search = GridSearchCV(SVC(), 
                                   param_grid = hyperparams, 
                                   cv = cross_valid)
        grid_search.fit(data, labels)
        return grid_search.best_params_, grid_search.best_score_
    
    PARAM, score = CV_SVM(data=data, labels=labels, K=K)
    
    clf = SVC(C = PARAM['C'], gamma = PARAM['gamma'])
    clf.fit(data, labels)
    pred_labels = clf.predict(data) 
    
    if verbose != -1:
        print('Done.\nAccuracy: %f' % accuracy_score(labels, pred_labels))
    
    return(pred_labels, accuracy_score(labels, pred_labels))




def LogReg(data, labels, K, verbose=1):
    '''
    Logistic Regression: l1-regularization was used to introduce sparsity.
    Regularization parameter C was chosen in the range of [10e−2, 10e3].
    '''
    def CV_LogReg(data, labels, K):
        # perform cross-validation for Logistic Regression classification
        hyperparams = {'C' : [0.01, 0.1, 1, 10, 100, 1000]}
        cross_valid = StratifiedKFold(n_splits = K, 
                                      shuffle = True, random_state = 0)
        grid_search = GridSearchCV(LogisticRegression(penalty = 'l1',
                                                      solver='liblinear'), 
                                   param_grid = hyperparams, 
                                   cv = cross_valid)
        grid_search.fit(data, labels)
        return grid_search.best_params_, grid_search.best_score_
    
    PARAM, score = CV_LogReg(data=data, labels=labels, K=K)
    
    clf = LogisticRegression(penalty = 'l1',
                             C = PARAM['C'], 
                             solver='liblinear')
    clf.fit(data, labels)
    pred_labels = clf.predict(data) 
    
    if verbose != -1:
        print('Done.\nAccuracy: %f' % accuracy_score(labels, pred_labels))
    
    return(pred_labels, accuracy_score(labels, pred_labels))




def RandForest(data, labels, K, DEPTH, verbose=1):
    '''
    random forest 
    The number of RandForest estimators is chosen within the set of {5, 10, 20, 50, 100, 500, 1000}.
    '''
    def CV_RandForest(data, labels, K):
        # perform cross-validation for RF classification
        hyperparams = {'n_estimators' : [5, 10, 20, 50, 100, 500, 1000]}
        cross_valid = StratifiedKFold(n_splits = K, 
                                      shuffle = True, random_state = 0)
        grid_search = GridSearchCV(RandomForestClassifier(max_depth = DEPTH), 
                                   param_grid = hyperparams, 
                                   cv = cross_valid)
        grid_search.fit(data, labels)
        return grid_search.best_params_, grid_search.best_score_
    
    PARAM, score = CV_RandForest(data=data, labels=labels, K=K)
    
    clf = RandomForestClassifier(n_estimators = PARAM['n_estimators'], 
                                 max_depth = DEPTH)
    clf.fit(data, labels)
    pred_labels = clf.predict(data) 
    
    if verbose != -1:
        print('Done.\nAccuracy: %f' % accuracy_score(labels, pred_labels))
    
    return(pred_labels, accuracy_score(labels, pred_labels))




pred_y, acc_score = SVM(data = np.reshape(X, (-1,3*64)), labels = y, K=3)
# pred_y, acc_score = LogReg(data = np.reshape(X, (-1,3*64)), labels = y, K=3)
# pred_y, acc_score = RandForest(data = np.reshape(X, (-1,3*64)), labels = y, K=3, DEPTH = 2)