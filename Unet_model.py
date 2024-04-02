
from keras.layers import *
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from Attention import CBAM_attention

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D, concatenate, BatchNormalization, \
    Activation, Add
from tensorflow.keras.models import Model


def conv_block(input_tensor, num_filters):
    """卷积块: 两个3x3卷积层，每层后面跟一个BatchNormalization层"""
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_tensor)
    x = Conv2D(num_filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    return x


def upsample_and_add(x, y):
    """上采样并加：上采样x到y的尺寸，然后与y相加前通过1x1卷积调整通道数"""
    # 上采样
    up = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    # 使用1x1卷积减少通道数以匹配y
    up = Conv2D(filters=y.shape[-1], kernel_size=(1, 1), padding='same')(up)
    # 执行加法
    return Add()([up, y])



def unet_three_band_with_fpn(pretrained_weights=None,input_size=(256, 256, 3)):
    inputs = Input(input_size)
    # 编码器
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = conv_block(pool4, 1024)
    # 自顶向下路径和横向连接（FPN结构）
    up6 = upsample_and_add(conv5, conv4)
    conv6 = conv_block(up6, 512)
    up7 = upsample_and_add(conv6, conv3)
    conv7 = conv_block(up7, 256)
    up8 = upsample_and_add(conv7, conv2)
    conv8 = conv_block(up8, 128)
    up9 = upsample_and_add(conv8, conv1)
    conv9 = conv_block(up9, 64)
    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
def unet_three_band_with_fpn_CBAM_Encoder(pretrained_weights=None,input_size=(256, 256, 3)):
    inputs = Input(input_size)
    # 编码器
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = CBAM_attention(pool1) + pool1
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = CBAM_attention(pool2) + pool2
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = CBAM_attention(pool3) + pool3
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = CBAM_attention(pool4) + pool4
    conv5 = conv_block(pool4, 1024)
    # 自顶向下路径和横向连接（FPN结构）
    up6 = upsample_and_add(conv5, conv4)
    conv6 = conv_block(up6, 512)
    up7 = upsample_and_add(conv6, conv3)
    conv7 = conv_block(up7, 256)
    up8 = upsample_and_add(conv7, conv2)
    conv8 = conv_block(up8, 128)
    up9 = upsample_and_add(conv8, conv1)
    conv9 = conv_block(up9, 64)
    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


def unet_three_band(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    # 编码器
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 中间层
    conv5 = conv_block(pool4, 1024)
    #解码器
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(512, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = conv_block(merge6, 512)
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = conv_block(merge7, 256)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = conv_block(merge8, 128)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv_block(merge9, 64)
    # 输出层
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model

def unet_three_band_attention(pretrained_weights=None, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    # 编码器
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = CBAM_attention(pool1) + pool1
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = CBAM_attention(pool2) + pool2
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = CBAM_attention(pool3) + pool3
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = CBAM_attention(pool4) + pool4
    conv5 = conv_block(pool4, 1024)
    #解码器
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(512, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = conv_block(merge6, 512)
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = conv_block(merge7, 256)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = conv_block(merge8, 128)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv_block(merge9, 64)
    # 输出层
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model



def unet_ten_band(pretrained_weights=None, input_size=(256, 256, 10)):
    inputs = Input(input_size)
    # 编码器
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = conv_block(pool3, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    # 中间层
    conv5 = conv_block(pool4, 1024)
    #解码器
    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(512, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up6)
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = conv_block(merge6, 512)
    up7 = UpSampling2D(size=(2, 2))(conv6)
    up7 = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = conv_block(merge7, 256)
    up8 = UpSampling2D(size=(2, 2))(conv7)
    up8 = Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = conv_block(merge8, 128)
    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = conv_block(merge9, 64)
    # 输出层
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=conv10)
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