import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from eeg_loader_ha import *
from config import *
from train_config import ResultDir

input_config = dict(window_len=80, overlap=0, mesh=False, one_hot=False)

train_index_t, train_index_r, valid_index_t, valid_index_r = split_ha(ResultDir)
train_X, train_y, train_D = load_data_from_file(train_index_r, train_index_t)

hyp = hyperalignment()
mappers = hyp.train(train_X)

train_X = [mappers[i].transform(x) for i, x in enumerate(train_X)]

fname = os.path.join(ResultDir, 'mappers.pkl')
with open(fname, 'wb') as f:
    pickle.dump(mappers, f)

train_X, train_y = get_data(train_X, train_y, train_D, **input_config)

valid_X, valid_y, valid_D = load_data_from_file(valid_index_r, valid_index_t)
valid_X = [mappers[i].transform(x) for i, x in enumerate(valid_X)]
valid_X, valid_y = get_data(valid_X, valid_y, valid_D, **input_config)

# average data across window length
train_X = train_X.mean(axis=1)
valid_X = valid_X.mean(axis=1)

# train svm model
clf = svm.SVC(gamma='scale')
clf.fit(train_X, train_y)  

# predict validation set 
valid_pre_y = clf.predict(valid_X) 
val_acc = accuracy_score(valid_pre_y, valid_y)
print('valid accuracy: ', val_acc)