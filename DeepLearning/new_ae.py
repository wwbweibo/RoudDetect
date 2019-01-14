import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2DTranspose, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt


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
    # model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu', name="encoder_output"))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Reshape((2, 2, 128)))
# ------------------------------
    model.add(Dense(128, activation='relu'))
    model.add(Reshape((2,2,32)))

    model.add(Conv2DTranspose(128, (3, 3), strides=2,
                              activation='relu', padding='same'))
    model.add(Conv2DTranspose(64, (3, 3), strides=2,
                              activation='relu', padding='same'))
    model.add(Conv2DTranspose(32, (3, 3), strides=2,
                              activation='relu', padding='same'))
    model.add(Conv2D(1, (3, 3), padding='same', activation='relu'))
    model.compile(optimizer='sgd', loss='mae')
    return model


def train():
    model = gen_model()
    # model.load_weights("models/ae_0112.h5")
    batch = np.load('images/x.npy')
    data = np.load('images/16px_image_x.npy')
    data = np.reshape(data, (40000, 16,16,1))
    data = np.concatenate((batch, data))
    model.fit(x=data, y=data, epochs=100, batch_size=40, validation_split=0.1)
    model.save_weights("models/ae_0112_new.h5")
    idx = np.random.randint(0, data.shape[0], size=8)
    x = data[idx]
    x = np.reshape(x, (8, 16, 16, 1))
    ret = model.predict(x)
    for i in range(4):
        for j in range(2):
            plt.subplot(4, 4, (i*4+j*2+1))
            plt.imshow(x[i*2+j, :, :, 0], cmap='gray')
            plt.subplot(4, 4, (i*4+j*2+2))
            plt.imshow(ret[i*2+j, :, :, 0], cmap='gray')
    plt.show()


if __name__ == "__main__":
    train()
