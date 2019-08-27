# ResNetEEG

# Google Cloud Tutorial
## Create Virtual Machine (with GPU)
* Click  [here](https://cloud.google.com/compute/docs/gpus/add-gpus)  to setup a VM instance with GPU.
* Issue:  quota 'gpus_all_regions' exceeded. limit 1.0 globally. [Trouble Shooting](https://stackoverflow.com/questions/53415180/gcp-error-quota-gpus-all-regions-exceeded-limit-0-0-globally)
* Settings (Recommended)
	* n1-standard-8 (8 vCPUs, 30 GB memory) 
	* GPUs (1 x NVIDIA Tesla K80)
	* System: Ubuntu1604 (Come with python3.5.2 & python2.7)
	* Disk Size 200
	* SSH Keys: Block project-wide SSH keys
	* Cloud API access scopes: Allow full access to all Cloud APIs

## Resize Disk
* Click [here](https://cloud.google.com/compute/docs/disks/add-persistent-disk#resize_partitions) for more details.

# Environment Setup
## Python Installation
* Option 1: Anaconda
* Option 2: Stand alone Python (Pip needs to be installed)

## Pip Installation
* sudo apt-get install python3-pip
* Trouble Shooting: update apt-get

## CUDA & cudnn Installation
[Install CUDA 9.0 and cuDNN 7.0 for TensorFlow/PyTorch (GPU) on Ubuntu 16.04](https://medium.com/repro-repo/install-cuda-and-cudnn-for-tensorflow-gpu-on-ubuntu-79306e4ac04e)

## Jupyter Notebook
* Jupyter/IPython installation 
	* pip install python/jupyter notebook
	* Trouble Shooting: update pip
*  [Installing Jupyter Notebook on Google Cloud](https://medium.com/@kn.maragatham09/installing-jupyter-notebook-on-google-cloud-11979e40cd10)

## Virtual Environment
1. Install virtualenv
pip install virtualenv
2. Install virtual environment with python3
virtualenv --python=_usr_bin_python3 <path_to_your_env_name/>
3. Install Jupyter kernel
		* enter virtualenv source _env_name_bin/activate
		* python -m ipykernel install --user --name env_name --display-name "env_name(for display only)"
    
### Create GCP configuration for new projects
[Configuring Google Cloud SDK for multiple projects](https://www.the-swamp.info/blog/configuring-gcloud-multiple-projects/)

