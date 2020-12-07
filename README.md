# ResNet-LSTM Motor Imagery Classification

## Getting Started
Training and testing the models require extensive GPU usage. AWS and Google cloud are recommended. Read [instructions](./gcloud_tutorial/README.md) to get started with GCP service.

### Prerequisites

* Python 3.5
* CUDA 9.0 and cudnn 7 (Installation guide [Install CUDA 9.0 and cuDNN 7.0 for TensorFlow/PyTorch (GPU) on Ubuntu 16.04](https://medium.com/repro-repo/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e))
```
bash cuda_cudnn_installer.sh
```

### Installing
* Create virtual environment for the project (recommended)
* Pip install all the required packages

```
pip install -r requirements.txt
pip install hypertools
```
(Optional) To disable plot function of hypertools, go to the directory where it is installed, i.e. ./virtual_env/lib/python3.5/site-packages/hypertools, and comment 'from .plot.plot import plot' to speed up loading process.

### Dataset
* S89 is rejected
* Get data from [physionet](https://www.physionet.org/content/eegmmidb/1.0.0/)
```
bash download_data.sh
```

* S88, S92, S100 have different sampling rates. Will consider downsampling all the data to the same sampling frequency.
* S89 is damaged.

* Split training/validation/test set, if there is no configuration file uner './config'. 
```
cd ResNetEEG/src
python select_sample.py
```
## HyperAlignment (optional)


## Training

1. Modify configuration by changing variables in ./src/generate_config.py.
2. If train from scratch
```
bash eeg_train.sh
```
3. Resume Training
```
python eeg_BCI2000_train.py -p yyyymmdd/yymmddhh -r r
```

## Test

1. Test the model on the test set

```
python eeg_BCI2000_test.py -p yyyymmdd/yymmddhh (-m b)
```

## Evaluation
### Usage 
```
from evaluation import *
# plot single confusion matrix
model_index = '20190822'
plot_single_cm(model_index)

# plot confusion matrices for  multiple models
model_indices = ['20190822', '2019xxx']
evaluation_compare(model_indices)



```
## Result
* SVM train 39.49% test 25.69%

## Deployment

OnlineTest
