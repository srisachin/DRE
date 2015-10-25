import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import data
from skimage import io
from skimage.measure import label
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.color import rgb2gray
from skimage.segmentation import clear_border
from skimage.transform import resize
from skimage.exposure import equalize_hist
from skimage.morphology import disk
from skimage.filters.rank import maximum
import PIL
from os import listdir
from os.path import isfile, join
import time


start = time.time()
#mypath = '/Users/Kevin/Desktop/sample/'

mypath = '/home/ubuntu/Kaggle/test/'

onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]


for file in onlyfiles[0:len(onlyfiles)]:
    try:
        im = data.imread(mypath+file)

        imGray = rgb2gray(im)

    #io.imshow(imGray)
    #io.show()

        imSegmented = imGray > 0.05

        imSegmented = maximum(imSegmented, disk(10))

    #io.imshow(imSegmented)
    #io.show()

        label_image = label(imSegmented)

        maxArea = 1

        dimensionsX, dimensionsY, z = im.shape

        minY, minX, z = im.shape
        maxX = 0
        maxY = 0
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            area = (maxr - minr)*(maxc - minc)
            if area < (dimensionsX * dimensionsY) / 4:
                continue
            if minr < minY:
                minY = minr
            if minc < minX:
                minX = minc
            if maxr > maxY:
                maxY = maxr
            if maxc > maxX:
                maxX = maxc
            if area > maxArea:
                imCropped = imGray[float(minY):float(maxY),float(minX):float(maxX)]
                imResized = resize(imCropped, [512, 512])
                imHist = equalize_hist(imResized)
                #io.imshow(imHist)
                #io.show()
                io.imsave('/home/ubuntu/Kaggle/testprocessed/'+file,imHist)
    except KeyError:
        continue
               
    

elapsed = (time.time() - start)
print(elapsed)
