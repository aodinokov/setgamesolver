#!/bin/bash

rm -rf wrk
mkdir -p wrk
unzip ../dataset/classification/classification_from_detection.zip -d wrk
unzip ../dataset/classification/classification_from_assets.zip -d wrk
unzip ../dataset/classification/classification_from_set-game-model.zip -d wrk

# mix all together
cd wrk
rm -rf mixed || true
mv classification_from_set-game-model/ mixed
cd classification_from_detection/; for i in $(find ./ -name *.jpg ); do mv $i ../mixed/$i; done; cd ..
cd classification_from_assets/; for i in $(find ./ -name *.jpg ); do mv $i ../mixed/$i; done; cd ..
rm -rf classification_from_detection
rm -rf classification_from_assets
