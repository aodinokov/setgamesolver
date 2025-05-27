#!/bin/bash

rm -rf wrk1
mkdir -p wrk1
unzip ../dataset/detection/annotations.zip -d wrk1
python3 _regenerate_class_from_det.py
# zip back to the similar arhive
cd wrk1; zip -1 -r classification_from_detection.zip classification_from_detection/; cd ..;
