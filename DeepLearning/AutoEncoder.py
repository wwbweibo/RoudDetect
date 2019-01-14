import os
import cv2
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation
from keras.layers import Conv2D, Input, UpSampling2D, MaxPooling2D, Dropout, Dense, Reshape, Flatten
from keras.layers.merge import Add
from keras.models import Model
from matplotlib import pyplot as plt
from utils.DataLoader import load_gaps


class AutoEncoder:
    def __init__(self, input_shape=(64, 64, 1), pre_trained=False, weights_path=None):
        """
        this is a AutoEncoder that will use as the first layers of the model
        we merge the encoder and the decoder hidden layers' input as the bbox regression input 
        and use the encoder output as the classifer regression input
        @param pre_trained: is this model pre-trained
        @param weigth_path: the pre-trained model weights path. if the ```pre_trained``` is true, must have a value
        """
        self.encoder_input_shape = input_shape
        self.decoder_input_shape = (
            int(input_shape[0] / 8), int(input_shape[1] / 8), 32)
        self.encoder = self.__gen_encoder__()
        self.decoder = self.__gen_decoder__()
        self.auto_encoder = self.__gen_autoencoder__()
        if pre_trained:
            if weights_path is None:
                raise ValueError(
                    "weights_path must has a value if pre-trained is true")
            if os.path.exists(weights_path + "_encoder.h5"):
                self.encoder.load_weights(weights_path + "_encoder.h5")
                self.decoder.load_weights(weights_path + "_decoder.h5")
                self.auto_encoder.load_weights(
                    weights_path + "_autoencoder.h5")
            else:
                raise ValueError("path is not exist")

    def __encoder_conv_block__(self, ipt, filter, stage, kernel_size=(3, 3)):
        x = ipt
        layer_name = "encoder_stage%d_%s_%d"
        x = Conv2D(filter, kernel_size, padding='same',
                   name=layer_name % (stage, 'conv', 0))(x)
        x = Conv2D(filter, kernel_size, padding='same',
                   name=layer_name % (stage, 'conv', 1))(x)
        x = BatchNormalization(momentum=0.8)(x)
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
        encoder_input = Input(
            shape=self.encoder_input_shape, name='encoder_input')

        stage1 = self.__encoder_conv_block__(encoder_input, 32, 1)
        stage1 = MaxPooling2D((2, 2))(stage1)
        stage1 = Dropout(0.25)(stage1)

        stage2 = self.__encoder_conv_block__(stage1, 32, 2)
        stage2 = MaxPooling2D((2, 2))(stage2)
        stage2 = Dropout(0.25)(stage2)

        stage3 = self.__encoder_conv_block__(stage2, 32, 3)
        stage3 = MaxPooling2D((2, 2))(stage3)
        stage3 = Dropout(0.25)(stage3)

        x = Flatten()(stage3)

        encoder = Model(encoder_input, x)
        encoder.summary()
        return encoder

    def __gen_decoder__(self):
        decoder_input = Input(shape=(self.decoder_input_shape[0] * self.decoder_input_shape[1]
                                     * self.decoder_input_shape[2],), name='decoder_input')
        # x = Dense(self.decoder_input_shape[0] * self.decoder_input_shape[1]
        #           * self.decoder_input_shape[2], activation='relu')(decoder_input)
        x = Reshape(self.decoder_input_shape)(decoder_input)

        stage1 = self.__decoder_conv_block__(x, 32, 1)
        stage1 = UpSampling2D((2, 2))(stage1)
        # 16
        stage2 = self.__decoder_conv_block__(stage1, 32, 2)
        stage2 = UpSampling2D((2, 2))(stage2)

        stage3 = self.__decoder_conv_block__(stage2, 32, 3)
        stage3 = UpSampling2D((2, 2))(stage3)

        output = Conv2D(1, (1, 1), activation='sigmoid',
                        padding='same', name='decoder_output')(stage3)

        decoder = Model(decoder_input, output)
        decoder.summary()
        return decoder

    def __gen_autoencoder__(self):
        autoencoder_input = Input(
            shape=self.encoder_input_shape, name='autoencoder_input')
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
        data = np.load('images/64px_image_x.npy')
        data = np.reshape(data, (40000, 64, 64, 1))
        self.auto_encoder.fit(x=data, y=data,
                              batch_size=16, epochs=epochs,
                              callbacks=[TensorBoard(log_dir='./logs')],
                              validation_split=0.1)

    def test_model(self):
        # im = cv2.imread('../test_img/th1.jpg', cv2.IMREAD_GRAYSCALE)

        # im = im.astype(np.float)
        # im /= 255
        # ret = self.auto_encoder.predict(
        #     [im.reshape((1, im.shape[0], im.shape[1], 1))])
        # plt.imshow(ret[0, :, :, 0], cmap='gray')
        # plt.show()

        data = np.load('images/64px_image_x.npy')
        idx = np.random.randint(0, 40000, size=8)
        x = data[idx]
        x = np.reshape(x, (8, 64, 64, 1))
        ret = self.auto_encoder.predict(x)
        for i in range(4):
            for j in range(2):
                plt.subplot(4, 4, (i*4+j*2+1))
                plt.imshow(x[i*2+j, :, :, 0], cmap='gray')
                plt.subplot(4, 4, (i*4+j*2+2))
                plt.imshow(ret[i*2+j, :, :, 0], cmap='gray')

        plt.show()

    def get_models(self):
        """
        return the encoder, decoder and autoencoder
        """
        return self.encoder, self.decoder, self.auto_encoder

    def save_weights(self, weights_name):
        self.encoder.save_weights(weights_name + "_encoder.h5")
        self.decoder.save_weights(weights_name + "_decoder.h5")
        self.auto_encoder.save_weights(weights_name + "_autoencoder.h5")


def main(weights_name, epochs=100, pre_trained=False):
    Auto_Encoder = AutoEncoder(
        pre_trained=pre_trained, weights_path=weights_name)
    Auto_Encoder.train(epochs=epochs)
    Auto_Encoder.test_model()
    Auto_Encoder.save_weights(weights_name)


if __name__ == '__main__':
    main("models/ae_64px", 2)
