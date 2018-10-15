import numpy as np
from keras.datasets import cifar10
from keras.layers import Dense, Flatten, Reshape
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from matplotlib import pyplot as plt

Input_layer = Input(shape=(None, None, 3))
conv1_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(Input_layer)
conv1_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1_1)
maxpool1 = MaxPooling2D((2, 2))(conv1_2)

conv2_1 = Conv2D(64, (3, 3), padding='same', activation='relu')(maxpool1)
conv2_2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv2_1)
maxpool2 = MaxPooling2D((2, 2))(conv2_2)
#
# conv3_1 = Conv2D(128, (3, 3), padding='same', activation='relu')(maxpool2)
# conv3_2 = Conv2D(128, (3, 3), padding='same', activation='relu')(conv3_1)
# maxpool3 = MaxPooling2D((2, 2))(conv3_2)
# conv4_1 = Conv2D(256, (3, 3), activation='relu', padding='same')(maxpool3)
# conv4_2 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4_1)

model = Model(inputs=Input_layer, outputs=maxpool2)

input2 = Input(shape=(None, None, 64))
dense = Dense(512, activation='relu')(input2)
reshape = Reshape((8, 8, 512))(dense)
flatten = Flatten()(reshape)
dense = Dense(512, activation='relu')(flatten)
dense = Dense(10, activation='softmax')(dense)
model2 = Model(inputs=input2, outputs=dense)
model2.summary()

input_merge = Input(shape=(None, None, 3))
t = model(input_merge)
output = model2(t)
model3 = Model(inputs=input_merge, outputs=output)
model3.compile(optimizer="adam", loss="categorical_crossentropy" )
model3.summary()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype(np.float32)
x_train /= 255

x_test = x_test.astype(np.float32)
x_test /= 255

y_train = to_categorical(y_train, 10)

model3.fit(x_train, y_train, 16, 5)
model.save("model.h5")
model2.save("model2.h5")
model3.save("model3.h5")

# model.load_weights("model.h5")
# model2.load_weights("model2.h5")
# model3.load_weights("model3.h5")


result = model3.predict(x_test[100:105])
print (result)
print (y_test[100:105])