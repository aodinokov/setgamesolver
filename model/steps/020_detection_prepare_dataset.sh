#!/bin/bash

source .venv/bin/activate

rm -rf wrk_d
mkdir -p wrk_d/images

# extract annotations
unzip ../dataset/detection/annotations.zip -d wrk_d

# creating two dir for training and testing
mkdir wrk_d/test_labels wrk_d/train_labels

# lists the files inside 'annotations' in a random order (not really random, by their hash value instead)
# Moves the first x labels (20% of the labels) to the testing dir: `test_labels`
test_subset_size=$(( $(ls wrk_d/annotations/* | wc -l) * 20 / 100 ))
ls wrk_d/annotations/* | sort -R | head -${test_subset_size} | xargs -I{} mv {} wrk_d/test_labels/

# Moves the rest of the labels ( 1096 labels ) to the training dir: `train_labels`
# ls wrk_d/annotations/* | xargs -I{} mv {} wrk_d/train_labels/
rm -rf wrk_d/train_labels && mv wrk_d/annotations wrk_d/train_labels

python3 << EOF
import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd

# if resized - use here scale
resize_scale=5

#adjusted from: https://github.com/datitran/raccoon_dataset
def xml_to_csv(path):
  classes_names = []
  xml_list = []

  for xml_file in glob.glob(path + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
      class_name = 'setgame-card' # forcing this isstead of the specific class from member[0].text
      classes_names.append(class_name)
      value = (root.find('filename').text,   
               int(float(root.find('size')[0].text)/resize_scale),
               int(float(root.find('size')[1].text)/resize_scale),
               class_name,
               int(float(member[4][0].text)/float(resize_scale)),
               int(float(member[4][1].text)/float(resize_scale)),
               int(float(member[4][2].text)/float(resize_scale)),
               int(float(member[4][3].text)/float(resize_scale)))
      xml_list.append(value)
  column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
  xml_df = pd.DataFrame(xml_list, columns=column_name) 
  classes_names = list(set(classes_names))
  classes_names.sort()
  return xml_df, classes_names

for label_path in ['wrk_d/train_labels', 'wrk_d/test_labels']:
  xml_df, classes = xml_to_csv(label_path)
  xml_df.to_csv(f'{label_path}.csv', index=None)
  print(f'Successfully converted {label_path} xml to csv.')

# generate label_map
label_map_path = os.path.join("wrk_d/label_map.pbtxt")
pbtxt_content = ""

for i, class_name in enumerate(classes):
    pbtxt_content = (
        pbtxt_content
        + "item {{\n    id: {0}\n    name: '{1}'\n}}\n\n".format(i + 1, class_name)
    )
pbtxt_content = pbtxt_content.strip()
with open(label_map_path, "w") as f:
    f.write(pbtxt_content)
    print('Successfully created label_map.pbtxt ')
EOF

## note: here is how classification from detection was created:
# import os
# import pandas as pd
# from PIL import Image

# files = ['train_labels.csv', 'test_labels.csv']
# prefix = 'images/'
# outputfolder = 'images_per_class/'

# def ensurePath(path):
# 	if not os.path.exists(path):
# 		os.makedirs(path)
# 		print("Created dir " + path)

# ensurePath(outputfolder)
# for f in files:
# 	df = pd.read_csv(f)
# 	for i,r in df.iterrows():
# 		img = Image.open(prefix+r['filename'])
# 		img2 = img.crop((r['xmin'], r['ymin'], r['xmax'], r['ymax']))
# 		dir = outputfolder + r['class']+'/'
# 		ensurePath(dir)
# 		file = dir + str(r['xmin']) +'x' + str(r['ymin']) +'x' + str(r['xmax']) +'x' + str(r['ymax']) + '-' + r['filename']
# 		img2.save(file)
# 		print("Created file " + file)

# resize pictures 5 times and save to a working dir
python3 << EOF
import glob
import os
from PIL import Image

sdir='../dataset/detection/images'
ddir='wrk_d/images'

for file in glob.glob(sdir + '/*.jpg'):
  im = Image.open(file)

  width, height = im.size
  im = im.resize((int(width/5), int(height/5)))

  dst=ddir + '/' + os.path.basename(file) 
  im.save(dst) #, quality=10)
  print("Saved " + file + " to " + dst)
EOF

# generate tfrecord
wget https://github.com/techzizou/Train-Object-Detection-Model-TF-2.x/raw/refs/heads/main/generate_tfrecord.py
python3 generate_tfrecord.py wrk_d/train_labels.csv wrk_d/label_map.pbtxt wrk_d/images/ wrk_d/train.record

rm generate_tfrecord.py