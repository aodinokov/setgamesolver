#!/bin/bash

rm -rf wrk
mkdir -p wrk
unzip ../dataset/classification/classification_from_detection.zip -d wrk
unzip ../dataset/classification/classification_from_assets.zip -d wrk
unzip ../dataset/classification/classification_from_set-game-model.zip -d wrk
