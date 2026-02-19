#!/bin/bash

# create virtual env with name .venv
python3 -m venv .venv
# use it and install needed pip libs (we need tf) 
source .venv/bin/activate
python3 -m pip install tensorflow[and-cuda]

# next python steps must be ran in venv (source .venv/bin/activate)