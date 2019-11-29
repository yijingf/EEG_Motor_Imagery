import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from matrix_display import *

data_dir = '../result/'
labels = ['rest', 'left_fist', 'right_fist', 'both_fist', 'both_feet']

def load_data(model):
    file_dir = os.path.join(data_dir, str(model), 'predicted_label_test.csv')
    res = pd.read_csv(file_dir)
    y_true, y_pred = res['y_true'], res['y_pred']
    return y_true, y_pred

def plot_single_cm(model):
    img = plt.figure()
    y_true, y_pred = load_data(model)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=labels, title='Confusion matrix for {}'.format(model))
    img.savefig(os.path.join(data_dir, model, 'Confustion_matrix'))

def evaluation_compare(models, n_col=4, save=True):
    cms_img = plt.figure(figsize=(20, 10))
    n = len(models)
    n_row = int(n/n_col)+1
    acc = []
    for i, model in enumerate(models):
        plt.subplot(n_row, n_col, i+1)
        y_true, y_pred = load_data(model)
        acc.append(accuracy_score(true_label, pred_label))
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, classes=labels, title='Confusion matrix for {}'.format(model))
    if save:
        plt.show()
        cms_img.savefig(os.path.join(data_dir, 'Confusion_matrix'))
    
        res = pd.DataFrame({'model':models, 'accuracy':acc})
        res.to_csv(os.path.join(data_dir, 'models_acc_compare.csv'), index=False)
    return