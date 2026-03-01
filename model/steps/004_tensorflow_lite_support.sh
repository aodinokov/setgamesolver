mkdir tmp
cd tmp

git clone --depth 1 https://github.com/tensorflow/tflite-support.git

# taken from https://github.com/google/flatbuffers/releases/tag/v25.12.19-2026-02-06-03fffb2
# correlates with what we have in pip list | grep flatbuffers
# flatbuffers              25.12.19
wget https://github.com/google/flatbuffers/releases/download/v25.12.19-2026-02-06-03fffb2/Linux.flatc.binary.clang++-18.zip
unzip Linux.flatc.binary.clang++-18.zip

# probably we need to have a copy of those???
wget https://github.com/tensorflow/tensorflow/raw/bd73701871af75539dd2f6d7fdba5660a8298caf/tensorflow/lite/schema/schema.fbs
wget https://github.com/tensorflow/tflite-support/raw/refs/heads/master/tensorflow_lite_support/metadata/metadata_schema.fbs

./flatc --version
./flatc --python --gen-object-api --gen-onefile schema.fbs
./flatc --python --gen-object-api --gen-onefile metadata_schema.fbs

# TODO: patch metadata.py
# TODO: copy all files it their places
