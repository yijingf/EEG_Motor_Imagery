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
```

### Dataset
* Get data from [physionet](https://www.physionet.org/content/eegmmidb/1.0.0/)
```
bash download_data.sh
```
* Split training/validation/test set, if there is no configuration file uner './config'. 
```
cd ResNetEEG/src
python select_sample.py
```

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


## Deployment

OnlineTest