import os
import tensorflow as tf
import parameters as params
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_discriminator():
    image_input = tf.keras.Input(shape=(params.image_height, params.image_width, 3), name='image_input')

    conv2d_1 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(image_input)
    conv2d_1 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_1)
    conv2d_1 = tf.keras.layers.LeakyReLU(alpha=0.3)(conv2d_1)

    conv2d_2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(conv2d_1)
    conv2d_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_2)
    conv2d_2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_2)

    conv2d_3 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(conv2d_2)
    conv2d_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_3)
    conv2d_3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_3)

    conv2d_4 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same')(conv2d_3)
    conv2d_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_4)
    conv2d_4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_4)

    flatten = tf.keras.layers.Flatten()(conv2d_4)
    dense_1 = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)

    model = tf.keras.Model(image_input, outputs=dense_1)
    return model


