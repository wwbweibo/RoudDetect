import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
import keras
from sklearn.cluster import KMeans
from keras.datasets import mnist
import numpy as np
import metrics
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn

from utils.DataLoader import load_gaps, load_stl10_data


def gen_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',
                     padding='same', input_shape=(16, 16, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    return model


def train():
    x, y = np.load('images/16px_image_x.npy'), np.load('images/16px_image_y.npy')
    y_ = keras.utils.to_categorical(y, 2)
    x = np.reshape(x, (40000,16,16,1))
    model = gen_model()
    model.fit(x, y_, batch_size=32, epochs=10, callbacks=[TensorBoard(
        log_dir='./logs/vgg', write_images=True)], validation_split=0.1)
    model.save_weights("models/like_vgg_3.h5")

    ret = model.predict(x)
    ret = ret.argmax(1)
    sns.set(font_scale=3)
    confusion_matrix = sklearn.metrics.confusion_matrix(y, ret)

    plt.figure(figsize=(16, 14))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20})
    plt.title("Confusion matrix", fontsize=30)
    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Clustering label', fontsize=25)
    plt.show()

train()
