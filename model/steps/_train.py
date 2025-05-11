import os

import numpy as np

import tensorflow as tf
assert tf.__version__.startswith('2')

# Set visible devices to only the second GPU (index 1)
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) >= 1:
    print(f"Using GPU")
else:
    print("Less than 1 GPUs available. Using CPU.")

import sys


from_folder='mixed'
export_dir='setgame-tflite'
args = sys.argv[1:]
if len(args) == 2:
  from_folder=args[0]
  export_dir=args[1]

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

import matplotlib.pyplot as plt

data = DataLoader.from_folder(from_folder)
train_data, test_data = data.split(0.95)
model = image_classifier.create(train_data, epochs=100)
loss, accuracy = model.evaluate(test_data)
model.export(export_dir=export_dir)
