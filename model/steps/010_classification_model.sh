
#!/bin/bash

source .venv/bin/activate

set -ex

# train from scratch => will produce checkpoints/default.classification.setgamemodel.weights.h5
# this is relativly quick process
python3 commands.py classification_train
cp checkpoints/default.classification.setgamemodel.weights.h5 checkpoints/trained.classification.setgamemodel.weights.h5

# fine-tune model. the current settings were able to do this for 995 epochs, setting 1300 just in case, anyway it has early stop
python3 commands.py classification_train --finetune --epoch_number 1300 --from_checkpoint_path checkpoints/trained.classification.setgamemodel.weights.h5
cp checkpoints/default.classification.setgamemodel.weights.h5 checkpoints/tuned.classification.setgamemodel.weights.h5

# export (to normal and tflite modes)
python3 commands.py classification_export --tflite --from_checkpoint_path checkpoints/tuned.classification.setgamemodel.weights.h5