from keras.layers import Conv2D, MaxPooling2D

def rpn(base_layers, num_anchors):
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='rpn_conv1')(base_layers)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='rpn_conv2')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1,1), activation='linear', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]