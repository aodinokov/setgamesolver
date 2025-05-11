#!/bin/bash

# do manually first
# sudo apt update
# sudo apt install software-properties-common
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt update
# sudo apt install python3.9 python3.9-venv
# sudo apt install libusb-1.0-0
# python3.9 -m venv ~/env001
# update ~.profile (path)

python3.9 $(which pip3) install numpy==1.23 tensorflow==2.8.0 tflite-model-maker==0.4.2 protobuf==3.20.3 packaging==20.9

# potentially
# python3.9  $(which pip3) install jupyterlab
#
# dependency for old cuda 11
# wget http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.2-0ubuntu2.1_amd64.deb
# sudo dpkg -i libtinfo5_6.2-0ubuntu2.1_amd64.deb
#
# install cuda 11
# wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
# sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
# sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
# sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get -y install cuda
#
# install libcudnn8 (download from https://developer.nvidia.com/rdp/cudnn-archive) - with hack (because it's for 2204 and we're doing 2404)
# sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
# sudo dpkg -i /var/cudnn-local-repo-ubuntu2204-8.9.7.29/libcudnn8_8.9.7.29-1+cuda11.8_amd64.deb
