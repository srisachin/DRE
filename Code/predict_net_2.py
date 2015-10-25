import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

caffe_root = '/home/ubuntu/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

MODEL_FILE = '/home/ubuntu/caffe/models/DRE/net2exp4/deploy.prototxt'
PRETRAINED = '/home/ubuntu/caffe/models/DRE/net2exp6/finexp1_iter_20000.caffemodel'
INPUT_FILE = '/home/ubuntu/Kaggle/ensemble_data/class_6.csv'
OUTPUT_FILE = '/home/ubuntu/Kaggle/ensemble_data/reg_6_6.csv'

import os
mean = []
caffe.set_mode_cpu()
mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1);

net = caffe.Classifier(MODEL_FILE, PRETRAINED)
net.set_raw_scale('data',255)
net.set_channel_swap('data',(2,1,0))
net.set_mean('data',np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'))

caffe.set_mode_gpu()

count = 0

f = open(OUTPUT_FILE, 'w')

#f.write("image,level\n")

with open(INPUT_FILE, 'rb') as csvfile:

  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:

    predictionVal = row[1]

    if row[1] == "1":
      IMAGE_FILE = "/home/ubuntu/Kaggle/test2/aa" + row[0]
      input_image = caffe.io.load_image(IMAGE_FILE)

      prediction = net.predict([input_image])
      prediction = int(round(prediction[0][0]))

      if prediction < 1:
        prediction = 1

      if prediction > 4:
        prediction = 4

      predictionVal = str(int(prediction))


    f.write(row[0] + "," + predictionVal + "\n")

    count = count + 1

    if count % 100 == 0:
      print count

f.close()
