import cv2
import keras.backend as K
import keras.utils
import numpy as np
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPool2D, UpSampling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Lambda
from keras.layers.merge import Add
from keras.layers.advanced_activations import ThresholdedReLU
from keras.losses import categorical_crossentropy
from keras.callbacks import TensorBoard
from keras.models import Model
from matplotlib import pyplot as plt
from gaps_dataset import gaps
from DataLoader import load_gaps_crack_images, load_gaps, load_gaps_cut
import time
import os


class AutoEncoder:
    def __init__(self, pre_trained=False, weights_path=None):
        """
        this is a AutoEncoder that will use as the first layers of the model
        we merge the encoder and the decoder hidden layers' input as the bbox regression input 
        and use the encoder output as the classifer regression input
        @param pre_trained: is this model pre-trained
        @param weigth_path: the pre-trained model weights path. if the ```pre_trained``` is true, must have a value
        """
        self.encoder = self.__gen_encoder__()
        self.decoder = self.__gen_decoder__()
        self.auto_encoder = self.__gen_autoencoder__()
        if pre_trained:
            if weights_path is None:
                raise ValueError(
                    "weights_path must has a value if pre-trained is true")
            if os.path.exists(weights_path+"_encoder.h5"):
                self.encoder.load_weights(weights_path+"_encoder.h5")
                self.decoder.load_weights(weights_path+"_decoder.h5")
                self.auto_encoder.load_weights(weights_path+"_autoencoder.h5")
            else:
                raise ValueError("path is not exist")

    def __encoder_conv_block__(self, ipt, filter, stage, conv_layer=2, kernel_size=(3, 3)):
        x = ipt
        layer_name = "encoder_stage%d_%s_%d"
        _x = Conv2D(filter, kernel_size, padding='same',
                    name=layer_name % (stage, 'conv', 0))(x)
        for i in range(1, conv_layer+1):
            x = Conv2D(filter, kernel_size, padding='same', kernel_initializer="normal",
                       name=layer_name % (stage, 'conv', i))(x)
            x = BatchNormalization()(x)
        x = Add()([x, _x])
        x = Activation('relu', name=layer_name % (stage, 'activation', 1))(x)
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
        stage1 = AveragePooling2D((2, 2))(stage1)
        # 32
        stage2 = self.__encoder_conv_block__(stage1, 16, 2, conv_layer=3)
        stage2 = AveragePooling2D((2, 2))(stage2)
        # 16
        stage3 = self.__encoder_conv_block__(stage2, 32, 3, conv_layer=3)
        stage3 = AveragePooling2D((2, 2))(stage3)
        # 8
        stage4 = self.__encoder_conv_block__(stage3, 32, 4, conv_layer=4)

        # output = Conv2D(16, (1, 1), padding='same',
        #                 activation='relu', name='encoder_outout')(stage5)

        output = Conv2D(4, (1, 1), padding='same',
                        activation='relu', name='encoder_outout')(stage4)
        encoder = Model(encoder_input, output)
        return encoder

    def __gen_decoder__(self):
        decoder_input = Input(shape=(None, None, 4), name='decoder_input')

        stage1 = self.__decoder_conv_block__(decoder_input, 32, 1)
        stage1 = UpSampling2D((2, 2))(stage1)
        # 16
        stage2 = self.__decoder_conv_block__(stage1, 32, 2)
        stage2 = UpSampling2D((2, 2))(stage2)
        # 32
        stage3 = self.__decoder_conv_block__(stage2, 16, 3)
        stage3 = UpSampling2D((2, 2))(stage3)

        stage4 = self.__decoder_conv_block__(stage3, 16, 4)
        output = Conv2D(1, (1, 1), activation='sigmoid',
                        padding='same', name='decoder_output')(stage4)

        decoder = Model(decoder_input, output)
        return decoder

    def __gen_autoencoder__(self):
        autoencoder_input = Input(
            shape=(None, None, 1), name='autoencoder_input')
        x = self.encoder(autoencoder_input)
        output = self.decoder(x)

        autoencoder = Model(autoencoder_input, output)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    def train(self, epochs=100, gaps_datasets=[0, 1], log_dir='./logs'):
        """
        this method will train the AutoEncoder.
        as default run 100 epochs in gaps dataset chunk 0 and 1. the training history will save in dir ```./logs```
        to see the history please using ```tensorboard```
        """
        # data = load_gaps_crack_images(gaps_datasets)
        # data = load_gaps_cut('images/cut_image/', 5500)
        data = np.load('images/cut_crack_image.npy')
        self.auto_encoder.fit(x=data, y=data,
                              batch_size=16, epochs=epochs,
                              callbacks=[TensorBoard(log_dir='./logs')],
                              validation_split=0.1)

    def test_model(self):
        # x = load_gaps_crack_images([0])
        # predict = self.auto_encoder.predict(np.asarray([x[10]]))
        # plt.subplot(1, 2, 1)
        # plt.imshow(x[0, :, :, 0], cmap='gray')
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.reshape(predict[0], (64, 64)), cmap='gray')
        # plt.show()
        # im = cv2.imread(
        #     '/media/weibo/Data/Code/RoadDetect/DeepLearning/images/images/train_0760.jpg', cv2.IMREAD_GRAYSCALE)
        im = cv2.imread('../test_img/th1.jpg', cv2.IMREAD_GRAYSCALE)

        im = im.astype(np.float)
        im /= 255
        ret = self.auto_encoder.predict(
            [im.reshape((1, im.shape[0], im.shape[1], 1))])
        plt.imshow(ret[0, :, :, 0], cmap='gray')
        plt.show()

    def get_models(self):
        """
        return the encoder, decoder and autoencoder
        """
        return self.encoder, self.decoder, self.auto_encoder

    def save_weights(self, weights_name):
        self.encoder.save_weights(weights_name+"_encoder.h5")
        self.decoder.save_weights(weights_name+"_decoder.h5")
        self.auto_encoder.save_weights(weights_name+"_autoencoder.h5")


def main(weights_name, epochs=100, pre_trained=False):
    Auto_Encoder = AutoEncoder(
        pre_trained=pre_trained, weights_path=weights_name)
    Auto_Encoder.train(epochs=epochs)
    Auto_Encoder.test_model()
    Auto_Encoder.save_weights(weights_name)


if __name__ == "__main__":
    main('models/train_1122_1', 20, False)
