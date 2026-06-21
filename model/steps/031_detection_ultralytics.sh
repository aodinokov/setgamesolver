#!/bin/bash

set -ex

source .venv/bin/activate
pip install ultralytics-opencv-headless

./convert_csv_to_yolo.py --output_dir=wrk_d_yolo --copy_images

yolo detect train data=wrk_d_yolo.yaml model=yolo26n.pt epochs=100 imgsz=640
# todo: https://docs.ultralytics.com/ru/modes/train#%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80%D1%8B-%D0%B8%D1%81%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F
# yolo train resume model=path/to/last.pt
# runs_dir=path/to/dir
# yolo detect val model=path/to/best.pt

yolo detect val model=../../runs/detect/train/weights/best.pt data=wrk_d_yolo.yaml

yolo export model=../../runs/detect/train/weights/best.pt format=tflite