#!/usr/bin/env python3
import glob
import xml.etree.ElementTree as ET
import os
import pandas as pd
from PIL import Image

def ensurePath(path):
	if not os.path.exists(path):
		os.makedirs(path)
		print("Created dir " + path)

def xml_to_folders(label_path, img_path, dst_path):
  classes_names = []
  xml_list = []

  for xml_file in glob.glob(label_path + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
      class_name = member[0].text
      classes_names.append(class_name)
      img = Image.open(os.path.join(img_path, root.find('filename').text))
      xmin = int(member[4][0].text)
      ymin = int(member[4][1].text)
      xmax = int(member[4][2].text)
      ymax = int(member[4][3].text)
      img2 = img.crop((xmin, ymin, xmax, ymax))
      dir = os.path.join(dst_path, class_name)
      ensurePath(dir)
      file = os.path.join(dir, str(xmin) +'x' + str(ymin) +'x' + str(xmax) +'x' + str(ymax) + '-' + root.find('filename').text)

      # update this vvvv if you need to make pics smaller
      scale = 1
      width, height = img2.size
      img2 = img2.resize((int(width/scale), int(height/scale)))

      img2.save(file)
      print("Created file " + file)
  classes_names = list(set(classes_names))
  classes_names.sort()
  return classes_names

# we will read everything from the original annotations
label_path = os.path.join(os.getcwd(), 'wrk1/annotations')
image_path = os.path.join(os.getcwd(), '../dataset/detection/images')
dst_path = os.path.join(os.getcwd(), 'wrk1/classification_from_detection')

xml_to_folders(label_path, image_path, dst_path)
