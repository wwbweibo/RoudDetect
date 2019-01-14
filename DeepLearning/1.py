from sklearn.cluster import KMeans
from keras.datasets import mnist
import numpy as np
import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn

# x, y = np.load('images/16px_image_x.npy'), np.load('images/16px_image_y.npy')

# x = np.reshape(x, (40000, 256))
# # 10 clusters
# # Runs in parallel 4 CPUs

# kmeans = KMeans(n_clusters=2, n_init=20)

# # Train K-Means.

# y_pred_kmeans = kmeans.fit_predict(x)

# # Evaluate the K-Means clustering accuracy.

# sns.set(font_scale=3)
# confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred_kmeans)

# plt.figure(figsize=(16, 14))
# sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
# plt.title("Confusion matrix", fontsize=30)
# plt.ylabel('True label', fontsize=25)
# plt.xlabel('Clustering label', fontsize=25)
# plt.show()

# print(metrics.acc(y, y_pred_kmeans))


import os
import cv2 as cv
from tqdm import tqdm

crack, no = [], []
f = open('images/DBCC_Training_Data_Set/train.txt')
for f in tqdm(f.readlines()):
    fn, label = f.split(' ')
    if int(label) == 0:
        im = cv.imread('images/DBCC_Training_Data_Set/train/'+fn,cv.IMREAD_GRAYSCALE)
        no.append(im)
    else:
        im = cv.imread('images/DBCC_Training_Data_Set/train/'+fn,cv.IMREAD_GRAYSCALE)
        crack.append(im)
f = open('images/DBCC_Training_Data_Set/val.txt')
for f in tqdm(f.readlines()):
    fn, label = f.split(' ')
    if int(label) == 0:
        im = cv.imread('images/DBCC_Training_Data_Set/val/'+fn,cv.IMREAD_GRAYSCALE)
        no.append(im)
    else:
        im = cv.imread('images/DBCC_Training_Data_Set/val/'+fn,cv.IMREAD_GRAYSCALE)
        crack.append(im)

crack = np.asarray(crack)
no = np.asarray(no)
data = np.concatenate((crack, no))
data = np.reshape(data,(len(crack) + len(no), 16, 16, 1))
one = np.ones(len(crack),dtype=np.uint8)
zero = np.zeros(len(no),dtype=np.uint8)
y = np.concatenate((one,zero))

np.save('images/x.npy',data)
np.save('images/y.npy', y)
print(data.shape)
print(len(crack))
print(len(no))