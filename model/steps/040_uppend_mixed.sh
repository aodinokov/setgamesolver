#!/bin/bash

cd wrk
cd classification_from_detection/; for i in $(find ./ -name *.jpg ); do cp $i ../mixed/$i; done; cd ..
cd classification_from_assets/; for i in $(find ./ -name *.jpg ); do cp $i ../mixed/$i; done; cd ..
