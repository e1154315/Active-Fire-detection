
import numpy as np
from keras.models import *
from keras.layers import *
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from Attention import CBAM_attention



def unet_three_band_attention(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    normalize1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normalize1)
    pool1 = CBAM_attention(pool1) + pool1

    # 池化后做attention
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    normalize2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normalize2)
    pool2 = CBAM_attention(pool2) + pool2

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    normalize3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normalize3)
    pool3 = CBAM_attention(pool3) + pool3

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    normalize4 = BatchNormalization()(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(normalize4)
    pool4 = CBAM_attention(pool4) + pool4

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    normalize6 = BatchNormalization()(up6)
    merge6 = concatenate([normalize4, normalize6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    normalize7 = BatchNormalization()(up7)
    merge7 = concatenate([normalize3, normalize7], axis=3)

    # merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    normalize8 = BatchNormalization()(up8)

    merge8 = concatenate([normalize2, normalize8], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    normalize9 = BatchNormalization()(up9)

    merge9 = concatenate([normalize1, normalize9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.2)(conv9)  # , training=True)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # Sigmoid activation for binary output

    model = Model(inputs=inputs, outputs=conv10)

    # model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_loss, metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model



def unet_three_band(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    normalize1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normalize1)
    # 池化后做attention
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    normalize2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normalize2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    normalize3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normalize3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    normalize4 = BatchNormalization()(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(normalize4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    normalize6 = BatchNormalization()(up6)
    merge6 = concatenate([normalize4, normalize6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    normalize7 = BatchNormalization()(up7)
    merge7 = concatenate([normalize3, normalize7], axis=3)

    # merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    normalize8 = BatchNormalization()(up8)

    merge8 = concatenate([normalize2, normalize8], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    normalize9 = BatchNormalization()(up9)

    merge9 = concatenate([normalize1, normalize9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.2)(conv9)  # , training=True)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # Sigmoid activation for binary output

    model = Model(inputs=inputs, outputs=conv10)

    #model.compile(optimizer=Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-8),
    #              loss=dice_loss,
    #              metrics=[dice_coefficient, iou])

    # model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_loss, metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

def unet_ten_band(pretrained_weights=None, input_size=(256, 256, 10)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    normalize1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normalize1)
    # 池化后做attention
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    normalize2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normalize2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    normalize3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normalize3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    normalize4 = BatchNormalization()(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(normalize4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    normalize6 = BatchNormalization()(up6)
    merge6 = concatenate([normalize4, normalize6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    normalize7 = BatchNormalization()(up7)
    merge7 = concatenate([normalize3, normalize7], axis=3)

    # merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    normalize8 = BatchNormalization()(up8)

    merge8 = concatenate([normalize2, normalize8], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    normalize9 = BatchNormalization()(up9)

    merge9 = concatenate([normalize1, normalize9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.2)(conv9)  # , training=True)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # Sigmoid activation for binary output

    model = Model(inputs=inputs, outputs=conv10)

    #model.compile(optimizer=Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-8),
    #              loss=dice_loss,
    #              metrics=[dice_coefficient, iou])

    # model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_loss, metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
















def unet_one_band(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    normalize1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normalize1)
    # 池化后做attention
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    normalize2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normalize2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    normalize3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normalize3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    normalize4 = BatchNormalization()(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(normalize4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    normalize6 = BatchNormalization()(up6)
    merge6 = concatenate([normalize4, normalize6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    normalize7 = BatchNormalization()(up7)
    merge7 = concatenate([normalize3, normalize7], axis=3)

    # merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    normalize8 = BatchNormalization()(up8)

    merge8 = concatenate([normalize2, normalize8], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    normalize9 = BatchNormalization()(up9)

    merge9 = concatenate([normalize1, normalize9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.2)(conv9)  # , training=True)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # Sigmoid activation for binary output

    model = Model(inputs=inputs, outputs=conv10)

    #model.compile(optimizer=Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-8),
    #              loss=dice_loss,
    #              metrics=[dice_coefficient, iou])

    # model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_loss, metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def unet_two_band(pretrained_weights=None, input_size=(256, 256, 2)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    normalize1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normalize1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    normalize2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normalize2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    normalize3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normalize3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    normalize4 = BatchNormalization()(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(normalize4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    normalize6 = BatchNormalization()(up6)
    merge6 = concatenate([normalize4, normalize6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    normalize7 = BatchNormalization()(up7)
    merge7 = concatenate([normalize3, normalize7], axis=3)

    # merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    normalize8 = BatchNormalization()(up8)

    merge8 = concatenate([normalize2, normalize8], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    normalize9 = BatchNormalization()(up9)

    merge9 = concatenate([normalize1, normalize9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.2)(conv9)  # , training=True)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # Sigmoid activation for binary output

    model = Model(inputs=inputs, outputs=conv10)

    # model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_loss, metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def unet_two_band_attention(pretrained_weights=None, input_size=(256, 256, 2)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    normalize1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normalize1)
    pool1 = CBAM_attention(pool1) + pool1

    # 池化后做attention
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    normalize2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normalize2)
    pool2 = CBAM_attention(pool2) + pool2

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    normalize3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normalize3)
    pool3 = CBAM_attention(pool3) + pool3

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    normalize4 = BatchNormalization()(conv4)
    # drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(normalize4)
    pool4 = CBAM_attention(pool4) + pool4

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    normalize6 = BatchNormalization()(up6)
    merge6 = concatenate([normalize4, normalize6], axis=3)

    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    normalize7 = BatchNormalization()(up7)
    merge7 = concatenate([normalize3, normalize7], axis=3)

    # merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    normalize8 = BatchNormalization()(up8)

    merge8 = concatenate([normalize2, normalize8], axis=3)

    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    normalize9 = BatchNormalization()(up9)

    merge9 = concatenate([normalize1, normalize9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.2)(conv9)  # , training=True)

    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)  # Sigmoid activation for binary output

    model = Model(inputs=inputs, outputs=conv10)

    # model.compile(optimizer=Adam(learning_rate=1e-5), loss=dice_loss, metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model