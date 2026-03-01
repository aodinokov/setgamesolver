#!/bin/bash

# NOTE: this is optional - just to regenerate folder if needed 
# remove the 'tensorflow_lite_support/metadata' folder manually first if you want to regenerate
mkdir -p tmp
pushd tmp

git clone --depth 1 https://github.com/tensorflow/tflite-support.git

# need flatbuffers binary which will generate 2 files for us below. See https://flatbuffers.dev/ for more info about this lib
# see how flatbuffes are used to build tflite structre here: https://ai.google.dev/edge/litert/conversion/tensorflow/metadata

# flatc is taken from https://github.com/google/flatbuffers/releases/tag/v25.12.19-2026-02-06-03fffb2
# so it correlates with what we have in pip list | grep flatbuffers
# flatbuffers              25.12.19
wget https://github.com/google/flatbuffers/releases/download/v25.12.19-2026-02-06-03fffb2/Linux.flatc.binary.clang++-18.zip
unzip Linux.flatc.binary.clang++-18.zip

./flatc --version

# generate schema_generated.py from fbs file from tensorflow project
# scheme doesn't change often. the commit sha is latest
wget https://github.com/tensorflow/tensorflow/raw/bd73701871af75539dd2f6d7fdba5660a8298caf/tensorflow/lite/schema/schema.fbs
./flatc --python --gen-object-api --gen-onefile schema.fbs

# generate metadata_schema_generated.py
./flatc --python --gen-object-api --gen-onefile tflite-support/tensorflow_lite_support/metadata/metadata_schema.fbs

# patch metadata.py to make it work without cc lib (which is only used to detect min metdata version)
patch tflite-support/tensorflow_lite_support/metadata/python/metadata.py << 'EOF'
30,31c30,31
< from tensorflow_lite_support.metadata.cc.python import _pywrap_metadata_version
< from tensorflow_lite_support.metadata.flatbuffers_lib import _pywrap_flatbuffers
---
> # from tensorflow_lite_support.metadata.cc.python import _pywrap_metadata_version
> # from tensorflow_lite_support.metadata.flatbuffers_lib import _pywrap_flatbuffers
305,306c305,307
<     min_version = _pywrap_metadata_version.GetMinimumMetadataParserVersion(
<         bytes(metadata_buf))
---
>     # min_version = _pywrap_metadata_version.GetMinimumMetadataParserVersion(
>     #     bytes(metadata_buf))
>     min_version = "1.0.0" # hardcode. 1.2.1 wasn't recognized by studio, alternatevly we can comment our this completely
EOF

popd
mkdir -p tensorflow_lite_support/metadata/metadata_writers/

# move all needed files to their places
mv tmp/tflite-support/tensorflow_lite_support/metadata/python/*.py* tensorflow_lite_support/metadata/
mv tmp/tflite-support/tensorflow_lite_support/metadata/python/metadata_writers/*.py* tensorflow_lite_support/metadata/metadata_writers/

# move generated files with proper names
mv tmp/schema_generated.py tensorflow_lite_support/metadata/schema_py_generated.py
mv tmp/metadata_schema_generated.py tensorflow_lite_support/metadata/metadata_schema_py_generated.py

rm -rf tmp