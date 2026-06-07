#!/bin/bash

# create virtual env with name .venv
python3 -m venv .venv
# use it and install needed pip libs (we need tf) 
source .venv/bin/activate
python3 -m pip install tensorflow[and-cuda]
python3 -m pip install pandas

# mediapipe didn't work
# tflite-support - doesn't work on python 3.12
# pip install ai-edge-litert tflite-support didn't work as well

# next python steps must be ran in venv (source .venv/bin/activate)