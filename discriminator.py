import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import parameters as params


def build_discriminator():
    image_input = tf.keras.Input(shape=(params.image_height, params.image_width, 3), name='image_input')
    label_input = tf.keras.Input(shape=(params.num_classes,), name='label_input')

    conv2d_1 = tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same')(image_input)
    pool_1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(conv2d_1)
    conv2d_2 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(pool_1)
    pool_2 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(conv2d_2)

    # output dim set as square root of number of classes
    label_embedding = tf.keras.layers.Embedding(params.num_classes, 8)(label_input)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)
    label_embedding = tf.keras.layers.Dense(8 * 8 * 64, activation="relu")(label_embedding)
    label_embedding = tf.keras.layers.Reshape((8, 8, 64))(label_embedding)
    concatenated = tf.keras.layers.Concatenate()([pool_2, label_embedding])

    flatten = tf.keras.layers.Flatten()(concatenated)
    dense_1 = tf.keras.layers.Dense(128, activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(64, activation='relu')(dense_1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_2)
    model = tf.keras.Model(inputs=[image_input, label_input], outputs=output)
    return model


