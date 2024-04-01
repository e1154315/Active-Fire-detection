# (1) Channel Attention

from keras.layers import *
import tensorflow as tf



def channel_attention(inputs, ratio=0.25):
    '''ratio represents the multiplier for reducing the number of channels in the first fully connected layer'''

    channel = inputs.shape[-1]  # Get the number of channels in the input feature map

    # Apply global max-pooling and global average-pooling to the output feature map separately
    # [h,w,c] => [None,c]
    x_max = GlobalMaxPooling2D()(inputs)
    x_avg = GlobalAveragePooling2D()(inputs)

    # [None,c] => [1,1,c]
    x_max = Reshape([1,1,-1])(x_max)  # -1 automatically finds the channel dimension size
    x_avg = Reshape([1,1,-1])(x_avg)  # Alternatively, you can use the variable 'channel' instead of -1

    # Reduce the number of channels by 1/4 in the first fully connected layer, [1,1,c] => [1,1,c//4]
    x_max = Dense(channel*ratio)(x_max)
    x_avg = Dense(channel*ratio)(x_avg)

    # Apply ReLU activation
    x_max = Activation('relu')(x_max)
    x_avg = Activation('relu')(x_avg)

    # Increase the number of channels in the second fully connected layer, [1,1,c//4] => [1,1,c]
    x_max = Dense(channel)(x_max)
    x_avg = Dense(channel)(x_avg)

    # Sum the results, [1,1,c] + [1,1,c] => [1,1,c]
    x = Add()([x_max, x_avg])

    # Normalize the weights using sigmoid
    x = tf.nn.sigmoid(x)

    # Multiply the input feature map by the weight vector to assign weights to each channel
    x = Multiply()([inputs, x])  # [h,w,c] * [1,1,c] => [h,w,c]

    return x

# (2) Spatial Attention
def spatial_attention(inputs):

    # Perform max-pooling and average-pooling over the channel dimension [b,h,w,c] => [b,h,w,1]
    # Set keepdims=False to get [b,h,w,c] => [b,h,w]
    x_max = tf.reduce_max(inputs, axis=3, keepdims=True)  # Compute the maximum value over the channel dimension
    x_avg = tf.reduce_mean(inputs, axis=3, keepdims=True)  # 'axis' can also be -1

    # Stack the results over the channel dimension [b,h,w,2]
    x = concatenate([x_max, x_avg])

    # Adjust the channels using a 1*1 convolution [b,h,w,1]
    x = Conv2D(filters=1, kernel_size=(1,1), strides=1, padding='same')(x)

    # Normalize the weights using the sigmoid function
    x = tf.nn.sigmoid(x)

    # Multiply the input feature map by the weight vector
    x = Multiply()([inputs, x])

    return x

# (3) CBAM Attention
def CBAM_attention(inputs):

    # Apply channel attention first and then spatial attention
    x = channel_attention(inputs)
    x = spatial_attention(x)
    return x