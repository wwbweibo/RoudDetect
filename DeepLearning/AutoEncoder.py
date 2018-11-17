import cv2
import keras.backend as K
import keras.utils
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D, UpSampling2D
from keras.layers import BatchNormalization, Activation
from keras.layers.merge import Add
from keras.layers.advanced_activations import ThresholdedReLU
from keras.losses import categorical_crossentropy
from keras.callbacks import TensorBoard
from keras.models import Model
from matplotlib import pyplot as plt
from gaps_dataset import gaps
from DataLoader import load_gaps_crack_images


def encoder_conv_block(ipt, filter, stage, conv_layer=2, kernel_size=(3, 3)):
    x = ipt
    layer_name = "encoder_stage%d_%s_%d"
    for i in range(conv_layer):
        x = Conv2D(filter, kernel_size, padding='same',
                   name=layer_name % (stage, 'conv', i))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Activation('relu')(x)
    return x


def decoder_conv_block(ipt, filter, stage):
    x = ipt
    layer_name = "decoder_stage%d_%s_%d"
    x = Conv2D(filter, (3, 3), activation='relu', padding='same', name=layer_name%(stage, 'conv', 1))(x)
    x = Conv2D(filter, (3, 3), activation='relu', padding='same', name=layer_name%(stage, 'conv', 2))(x)
    return x

def gen_encoder():
    encoder_input = Input(shape=(None, None, 1), name='encoder_input')
    stage1 = encoder_conv_block(encoder_input, 32, 1)
    stage1 = MaxPool2D((2, 2))(stage1)

    stage2 = encoder_conv_block(stage1, 32, 2, conv_layer=3)
    stage2 = MaxPool2D((2, 2))(stage2)

    stage3 = encoder_conv_block(stage2, 64, 3, conv_layer=3)
    stage3 = MaxPool2D((2, 2))(stage3)

    stage4 = encoder_conv_block(stage3, 128, 4, conv_layer=4)

    output = Conv2D(32, (1, 1), padding='same', activation='relu', name='encoder_outout')(stage4)

    encoder = Model(encoder_input, output)
    encoder.summary()
    return encoder


def gen_decoder():
    decoder_input = Input(shape=(None, None, 32), name='decoder_input')

    stage1 = decoder_conv_block(decoder_input, 32, 1)
    stage1 = UpSampling2D((2, 2))(stage1)

    stage2 = decoder_conv_block(stage1, 32, 2)
    stage2 = UpSampling2D((2, 2))(stage2)

    stage3 = decoder_conv_block(stage2, 16, 3)
    stage3 = UpSampling2D((2, 2))(stage3)

    stage4 = decoder_conv_block(stage3, 16, 4)
    output = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='decoder_output')(stage4)

    decoder = Model(decoder_input, output)
    decoder.summary()
    return decoder


def gen_autoencoder(encoder, decoder):
    autoencoder_input = Input(shape=(None, None, 1), name='autoencoder_input')
    x = encoder(autoencoder_input)
    output = decoder(x)

    autoencoder = Model(autoencoder_input, output)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


def train(model, epochs=100):
    data = load_gaps_crack_images([0])
    model.fit(x=data, y=data, batch_size=16, epochs=epochs, callbacks=[TensorBoard(log_dir='./logs')])


def main():
    encoder = gen_encoder()
    decoder = gen_decoder()
    autoencoder = gen_autoencoder(encoder, decoder)
    train(autoencoder)
    encoder.save_weights('models/encoder.h5')
    decoder.save_weights('models/decoder.h5')
    autoencoder.save_weights('models/autoencoder.h5')


if __name__ == '__main__':
    main()