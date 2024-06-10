import os
import tensorflow as tf
import parameters as params
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_generator():
    noise_input = tf.keras.Input(shape=(params.noise_dim,), name='noise_input')

    dense_1 = tf.keras.layers.Dense(128 * 8 * 8, activation='relu')(noise_input)
    reshape = tf.keras.layers.Reshape((8, 8, 128))(dense_1)

    # 16x16
    conv_1 = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(reshape)
    normal_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_1)
    activation_1 = tf.keras.layers.ReLU()(normal_2)

    # 32x32
    conv_2 = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same')(activation_1)
    normal_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_2)
    activation_2 = tf.keras.layers.ReLU()(normal_3)

    # 64x64
    conv_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(activation_2)
    normal_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_3)
    activation_3 = tf.keras.layers.ReLU()(normal_4)

    # 128x128
    conv_4 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same')(activation_3)
    normal_5 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_4)
    activation_4 = tf.keras.layers.ReLU()(normal_5)

    output = tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(activation_4)

    model = tf.keras.Model(noise_input, outputs=output)
    return model
