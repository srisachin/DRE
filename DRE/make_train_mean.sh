#!/usr/bin/env sh
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

./build/tools/compute_image_mean /home/ubuntu/caffe/train3 /home/ubuntu/caffe/models/DRE/data/trainingmean.binaryproto

echo "Done."
