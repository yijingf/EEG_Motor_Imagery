import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from eeg_loader import *

# load data
subs = ['S001', 'S002', 'S003']
data_loader = DataLoader()
(train_X, train_y), (valid_X, valid_y), test_set = data_loader.load_train_val_test(subs, mesh=False)

# average data across window length
train_X = train_X.mean(axis=-1)
valid_X = valid_X.mean(axis=-1)

# train svm model
clf = svm.SVC(gamma='scale')
clf.fit(train_X, train_y)  

# predict validation set 
valid_pre_y = clf.predict(valid_X) 
val_acc = accuracy_score(valid_pre_y, valid_y)
print('valid accuracy: ', val_acc)


