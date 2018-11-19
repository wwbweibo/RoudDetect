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
from DataLoader import load_gaps_crack_images, load_gaps
import time
import os

class AutoEncoder:
    def __init__(self, weights_path):
        """
        this is a AutoEncoder that will use as the first layers of the model
        we merge the encoder and the decoder hidden layers' input as the bbox regression input 
        and use the encoder output as the classifer regression input
        @param weigth_path: the pre-trained model weights path.
        """
        self.encoder = self.__gen_encoder__()
        self.decoder = self.__gen_decoder__()
        self.auto_encoder = self.__gen_autoencoder__()
        if weights_path is None:
            raise("please input the weights path")
        if os.path.exists(weights_path): 
            self.encoder.load_weights(weights_path+"/encoder.h5")
            self.decoder.load_weights(weights_path+"/decoder.h5")
            self.encoder.load_weights(weights_path+"/autoencoder.h5")

    def __encoder_conv_block__(self, ipt, filter, stage, conv_layer=2, kernel_size=(3, 3)):
        x = ipt
        layer_name = "encoder_stage%d_%s_%d"
        for i in range(conv_layer):
            x = Conv2D(filter, kernel_size, padding='same',
                       name=layer_name % (stage, 'conv', i))(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Activation('relu')(x)
        return x

    def __decoder_conv_block__(self, ipt, filter, stage):
        x = ipt
        layer_name = "decoder_stage%d_%s_%d"
        x = Conv2D(filter, (3, 3), activation='relu', padding='same',
                   name=layer_name % (stage, 'conv', 1))(x)
        x = Conv2D(filter, (3, 3), activation='relu', padding='same',
                   name=layer_name % (stage, 'conv', 2))(x)
        return x

    def __gen_encoder__(self):
        encoder_input = Input(shape=(None, None, 1), name='encoder_input')
        stage1 = self.__encoder_conv_block__(encoder_input, 16, 1)
        stage1 = MaxPool2D((2, 2))(stage1)
        # 32
        stage2 = self.__encoder_conv_block__(stage1, 32, 2, conv_layer=3)
        stage2 = MaxPool2D((2, 2))(stage2)
        # 16
        stage3 = self.__encoder_conv_block__(stage2, 64, 3, conv_layer=3)
        stage3 = MaxPool2D((2, 2))(stage3)
        # 8
        stage4 = self.__encoder_conv_block__(stage3, 128, 4, conv_layer=4)
        stage4 = MaxPool2D((2, 2))(stage4)
        # 4
        stage5 = self.__encoder_conv_block__(stage4, 256, 5, conv_layer=4)

        output = Conv2D(16, (1, 1), padding='same',
                        activation='relu', name='encoder_outout')(stage5)

        encoder = Model(encoder_input, output)
        encoder.summary()
        return encoder

    def __gen_decoder__(self):
        decoder_input = Input(shape=(None, None, 16), name='decoder_input')

        stage0 = self.__decoder_conv_block__(decoder_input, 256, 0)
        stage0 = UpSampling2D((2, 2))(stage0)

        stage1 = self.__decoder_conv_block__(stage0, 128, 1)
        stage1 = UpSampling2D((2, 2))(stage1)
        # 16
        stage2 = self.__decoder_conv_block____(stage1, 64, 2)
        stage2 = UpSampling2D((2, 2))(stage2)
        # 32
        stage3 = self.__decoder_conv_block__(stage2, 32, 3)
        stage3 = UpSampling2D((2, 2))(stage3)

        stage4 = self.__decoder_conv_block__(stage3, 16, 4)
        output = Conv2D(1, (1, 1), activation='sigmoid',
                        padding='same', name='decoder_output')(stage4)

        decoder = Model(decoder_input, output)
        decoder.summary()
        return decoder

    def __gen_autoencoder__(self):
        autoencoder_input = Input(
            shape=(None, None, 1), name='autoencoder_input')
        x = self.encoder(autoencoder_input)
        output = self.decoder(x)

        autoencoder = Model(autoencoder_input, output)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train(self, epochs=100, gaps_datasets=[0,1], log_dir='./logs'):
        """
        this method will train the AutoEncoder.
        as default run 100 epochs in gaps dataset chunk 0 and 1. the training history will save in dir ```./logs```
        to see the history please using ```tensorboard```
        """
        data = load_gaps_crack_images(gaps_datasets)
        self.auto_encoder.fit(x=data[0:7000], y=data[0:7000],
                              batch_size=16, epochs=epochs,
                              callbacks=[TensorBoard(log_dir='./logs')],
                              validation_data=[data[7001:], data[7001:]])

    def test_model(self, model, weights_name):
        model.load_weights(weights_name)
        x, _ = load_gaps()
        predict = model.predict(np.asarray([x[10]]))
        plt.subplot(1, 2, 1)
        plt.imshow(x[0, :, :, 0], cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(np.reshape(predict[0], (64, 64)), cmap='gray')
        plt.show()
        im = cv2.imread(
            '/media/weibo/Data/Code/RoadDetect/test_img/timg.jpg', cv2.IMREAD_GRAYSCALE)
        # im = cv2.imread('E:/Code/RoadDetect/test_img/th1.jpg', cv2.IMREAD_GRAYSCALE)
        im = im.astype(np.float)
        im /= 255
        ret = model.predict([im.reshape((1, im.shape[0], im.shape[1], 1))])
        plt.imshow(ret[0, :, :, 0], cmap='gray')
        plt.show()

    def get_models(self):
        """
        return the encoder, decoder and autoencoder
        """
        return  self.encoder, self.decoder, self.auto_encoder
