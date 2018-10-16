from keras.layers import Input
from keras.applications import VGG16
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt

import keras.backend as K
import cv2 as cv
import numpy as np


vgg = VGG16(weights='imagenet', include_top=False)
vgg.summary()
im = cv.imread("test_img/test.png",cv.IMREAD_ANYCOLOR)
im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
im_arr = np.asarray(im)
im_arr = np.reshape(im, (1, im.shape[0], im.shape[1], 3))
im_arr = im_arr.astype(np.float)
im_arr /= 255
layer = K.function([vgg.layers[0].input], [vgg.layers[16].output])

f1 = layer([im_arr])[0]

for j in range(16):
    for i in range(16):
        data = f1[0,:,:,(j * 16)+ i]
        plt.subplot(4,4,i + 1)
        plt.imshow(data, cmap='gray')

    plt.show()