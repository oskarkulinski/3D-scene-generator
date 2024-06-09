import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import parameters as params


def build_generator():
    noise_input = tf.keras.Input(shape=(params.noise_dim,), name='noise_input')

    noise_reshape = tf.keras.layers.Dense(8192, activation='relu')(noise_input)
    noise_reshape = tf.keras.layers.Reshape((8, 8, 128))(noise_reshape)

    # 16x16
    conv_1 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, padding='same', strides=2)(noise_reshape)
    normal_1 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_1)
    activation_1 = tf.keras.layers.ReLU()(normal_1)

    # 32x32
    conv_2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(activation_1)
    normal_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_2)
    activation_2 = tf.keras.layers.ReLU()(normal_2)

      # 64x64
    conv_3 = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(activation_2)
    normal_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_3)
    activation_3 = tf.keras.layers.ReLU()(normal_3)

    upsample_4 = tf.keras.layers.UpSampling2D(size=(2, 2))(activation_3)  # 128x128
    conv_4 = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, padding='same')(upsample_4)
    normal_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_4)
    activation_4 = tf.keras.layers.ReLU()(normal_4)

    output = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, padding='same', activation='tanh')(activation_4)

    model = tf.keras.Model(inputs=noise_input, outputs=output)
    return model