from gaps_dataset import gaps
import numpy as np
import keras.utils
from stl10_input import read_all_images, read_labels

def load_gaps():
    x, y = gaps.load_chunk(0, datadir='./', debug_outputs=True)
    x = np.reshape(x, (x.shape[0], 64, 64, 1))
    for im, idx in zip(x, range(32000)):
        if np.min(im) < 0:
            x[idx] = im + np.abs(np.min(im))
    return x, y


def load_data():
    x = read_all_images('stl10_binary/train_X.bin')
    y = read_labels('stl10_binary/train_y.bin')
    y = y - 1
    y = keras.utils.to_categorical(y, num_classes=10)

    val_x = read_all_images('stl10_binary/test_X.bin')
    val_y = read_labels('stl10_binary/test_y.bin')

    val_y = val_y - 1
    val_y = keras.utils.to_categorical(val_y, 10)
    return (x, y), (val_x, val_y)


def load_gaps_crack_images(chunk_list):
    ret_lst = []
    for chunk in chunk_list:
        x, y = gaps.load_chunk(chunk, datadir='.', debug_outputs=True)
        # find the 1 label
        ret = np.where(y == 1)
        crack_x = x[np.where(y==1)]
        crack_x = np.reshape(crack_x, (crack_x.shape[0], 64, 64, 1))
        for im, idx in zip(crack_x, range(crack_x.shape[0])):
            if np.min(im) < 0:
                crack_x[idx] = im + np.abs(np.min(im))
        ret_lst.append(crack_x)
    ret = np.asarray(ret_lst)
    ret = np.reshape(ret, (-1, 64, 64, 1))
    return ret
    