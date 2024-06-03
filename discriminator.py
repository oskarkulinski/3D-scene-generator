import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import parameters as params


def build_discriminator():
    image_input = tf.keras.Input(shape=(params.image_height, params.image_width, 3), name='image_input')
    label_input = tf.keras.Input(shape=(1,), dtype='int32', name='label_input')

    label_embedding = tf.keras.layers.Embedding(params.num_classes, params.image_height * params.image_width)(label_input)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)
    label_embedding = tf.keras.layers.Reshape((params.image_height, params.image_width, 1))(label_embedding)

    concatenated = tf.keras.layers.Concatenate()([image_input, label_embedding])

    conv2d_1 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(concatenated)
    conv2d_1 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_1)
    conv2d_1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_1)

    conv2d_2 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same')(conv2d_1)
    conv2d_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_2)
    conv2d_2 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_2)

    conv2d_3 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same')(conv2d_2)
    conv2d_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_3)
    conv2d_3 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_3)

    conv2d_4 = tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, padding='same')(conv2d_3)
    conv2d_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv2d_4)
    conv2d_4 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2d_4)

    flatten = tf.keras.layers.Flatten()(conv2d_4)
    dense_1 = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)

    model = tf.keras.Model(inputs=[image_input, label_input], outputs=dense_1)
    return model



