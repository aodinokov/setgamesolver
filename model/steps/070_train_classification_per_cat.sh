#!/bin/bash

cd wrk
# number
python3 ../_train.py _cat0 setgame-tflite-number
## color
#python3 ../_train.py _cat1 setgame-tflite-color
# shading
python3 ../_train.py _cat2 setgame-tflite-shading
# shape
python3 ../_train.py _cat3 setgame-tflite-shape
# 3 categories without color
python3 ../_train.py _3cat setgame-tflite-number-shading-shape
