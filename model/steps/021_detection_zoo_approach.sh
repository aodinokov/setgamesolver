#!/bin/bash

sudo apt install -y protobuf-compiler

source .venv/bin/activate

# clone the tensorflow models on the colab cloud vm
# Originally I used , but I guess it's more predictable to work with tags:
git clone --q https://github.com/tensorflow/models.git
pushd models/research

git checkout 5b2b5aa91b670624bae39faaa3aa6ec5eb2ee059

# Compile protos.
protoc object_detection/protos/*.proto --python_out=.

# Install TensorFlow Object Detection API.
cp object_detection/packages/tf2/setup.py .
python -m pip install .


popd
# rm -rf models

# generate tfrecord
# wget https://github.com/techzizou/Train-Object-Detection-Model-TF-2.x/raw/refs/heads/main/generate_tfrecord.py
python3 generate_tfrecord.py wrk_d/train_labels.csv wrk_d/label_map.pbtxt wrk_d/images/ wrk_d/train.record
python3 generate_tfrecord.py wrk_d/test_labels.csv  wrk_d/label_map.pbtxt wrk_d/images/ wrk_d/test.record
# rm generate_tfrecord.py