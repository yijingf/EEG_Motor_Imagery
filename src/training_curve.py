""" 
Plot the training curve of accuracy and loss

Usage: python training_curve.py <model name>
<model name> is a folder name under ../result, which is usuall a time stamp.
For example, run 'python training_curve.py 20210110' to plot 
'../result/20210110/train_log.txt' and '../result/20210110/val_log.txt'

Authors: Yijing Feng, Xinhui Li
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

root_dir = '../result/'

def load_data(model, mode):
    filename = os.path.join(root_dir, model, '{}_log.txt'.format(mode))
    with open(filename, 'r') as f:
        data = f.read().splitlines()
        
    if mode == 'val':
        data = [i.split(', ') for i in data if 'Validation' in i]
    else:
        data = [i.split(',') for i in data]
    step, loss, acc = [i for i in zip(*data)]

    step = [int(i.split(': ')[-1]) for i in step]
    loss = [float(i.split(': ')[-1]) for i in loss]
    acc = [float(i.split(': ')[-1]) for i in acc]
    
    return step, loss, acc

def plot_data(steps, metrics, metric_type, filename, smooth_factor=1600):
    # Draw lines
    fig = plt.figure()
    for i, mode in enumerate(['Training', 'Validation']):
        step_new = np.linspace(min(steps[i]), max(steps[i]), int(max(steps[i])/smooth_factor))
        spl = make_interp_spline(steps[i], metrics[i], k=3) #BSpline object
        metric_smooth = spl(step_new)
        step_new = step_new / smooth_factor
        plt.plot(step_new, metric_smooth, label="{} {}".format(mode, metric_type))

    # Create plot
    plt.title("Learning Curve - {}".format(metric_type))
    plt.xlabel("Epochs"), plt.ylabel(metric_type), plt.legend(loc="best")
    plt.tight_layout()
    
    fig.savefig(filename)
    return
    
def main(model):
    train_step, train_loss, train_acc = load_data(model, 'train')
    val_step, val_loss, val_acc = load_data(model, 'val')
    
    filename = os.path.join(root_dir, model, 'LossCurve')
    plot_data([train_step, val_step], [train_loss, val_loss], 'Loss', filename)
    
    filename = os.path.join(root_dir, model, 'AccCurve')
    plot_data([train_step, val_step], [train_acc, val_acc], 'Accuracy', filename)
    return

if __name__ == "__main__":
    main(sys.argv[1])