#!/bin/bash

sudo apt install -y protobuf-compiler

source .venv/bin/activate
pip install tensorflow-estimator # ==2.12.0

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


# TODO: try to find a fresher net and also increase resolution to 640x640
# Download the pre-trained model ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz into the data folder & unzip it.
pushd wrk_d
wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
tar -xzvf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz 
popd
cp models/research/object_detection/configs/tf2/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config wrk_d/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config.orig
cp ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config wrk_d/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config
diff wrk_d/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config.orig wrk_d/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config

PIPELINE_CONFIG_PATH=wrk_d/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config
MODEL_DIR=wrk_d/training
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

#python model_main_tf2.py --pipeline_config_path=/mydrive/customTF2/data/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config --model_dir=/mydrive/customTF2/training --alsologtostderr

pushd models/research/object_detection
export TF_USE_LEGACY_KERAS=1
python3 model_main_tf2.py -- \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
popd