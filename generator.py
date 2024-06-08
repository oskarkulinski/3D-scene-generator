import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import parameters as params


def build_generator():
    noise_input = tf.keras.Input(shape=(params.noise_dim,), name='noise_input')

    noise_reshape = tf.keras.layers.Dense(8192, activation='relu')(noise_input)
    noise_reshape = tf.keras.layers.Reshape((8, 8, 128))(noise_reshape)

    upsample_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(noise_reshape)  # 16x16
    conv_1 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu')(upsample_1)
    normal_1 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_1)
    dropout_1 = tf.keras.layers.Dropout(0.25)(normal_1)

    upsample_2 = tf.keras.layers.UpSampling2D(size=(2, 2))(dropout_1)  # 32x32
    conv_2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, padding='same', activation='relu')(upsample_2)
    normal_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_2)
    dropout_2 = tf.keras.layers.Dropout(0.25)(normal_2)

    upsample_3 = tf.keras.layers.UpSampling2D(size=(2, 2))(dropout_2)  # 64x64
    conv_3 = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, padding='same', activation='relu')(upsample_3)
    normal_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_3)
    dropout_3 = tf.keras.layers.Dropout(0.25)(normal_3)

    upsample_4 = tf.keras.layers.UpSampling2D(size=(2, 2))(dropout_3)  # 128x128
    conv_4 = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, padding='same', activation='relu')(upsample_4)
    normal_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_4)
    dropout_4 = tf.keras.layers.Dropout(0.25)(normal_4)

    output = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, padding='same', activation='tanh')(dropout_4)

    model = tf.keras.Model(inputs=noise_input, outputs=output)
    return model