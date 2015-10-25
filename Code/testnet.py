import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#%matplotlib inline

# Make sure that caffe is on the python path:
caffe_root = '/home/ubuntu/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = '/home/ubuntu/caffe/models/DRE/net2exp4/deploy.prototxt'
PRETRAINED = '/home/ubuntu/caffe/models/DRE/net2exp4/exp1_iter_15000.caffemodel'
IMAGE_FILE = '/home/ubuntu/Kaggle/trainprocessed/a38890_right.jpeg'

import os
mean = []
caffe.set_mode_cpu()
mean=np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1);
#mean=np.fromfile('/home/ubuntu/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy');

#plt.imshow('/home/ubuntu/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
net = caffe.Classifier(MODEL_FILE, PRETRAINED)
net.set_raw_scale('data',255)
net.set_channel_swap('data',(2,1,0))
net.set_mean('data',np.load(caffe_root+'python/caffe/imagenet/ilsvrc_2012_mean.npy'))
#net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=mean,
#                       channel_swap=(2,1,0),
#                       raw_scale=255,
#                       image_dims=(256, 256))

input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
plt.savefig('input_image.jpeg')
plt.clf()

prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
print 'predicted class:', prediction[0]

plt.savefig('example_plot.jpeg')


prediction = net.predict([input_image], oversample=False)
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
print 'predicted class:', prediction[0]

# Resize the image to the standard (256, 256) and oversample net input sized crops.
input_oversampled = caffe.io.oversample([caffe.io.resize_image(input_image, net.image_dims)], net.crop_dims)
# 'data' is the input blob name in the model definition, so we preprocess for that input.
caffe_input = np.asarray([net.transformer.preprocess('data', in_) for in_ in input_oversampled])
# forward() takes keyword args for the input blobs with preprocessed input arrays.
#%timeit net.forward(data=caffe_input)

caffe.set_mode_gpu()

prediction = net.predict([input_image])
print 'prediction shape:', prediction[0].shape
plt.plot(prediction[0])
