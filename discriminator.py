import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import parameters as params


def build_discriminator():
    image_input = tf.keras.Input(shape=(256, 256, 3), name='image_input')
    label_input = tf.keras.Input(shape=(params.num_classes,), name='label_input')

    # output dim set as square root of number of classes
    label_embedding = tf.keras.layers.Embedding(params.num_classes, 8)(label_input)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)
    label_embedding = tf.keras.layers.Dense(256 * 256 * 3, activation="relu")(label_embedding)
    label_embedding = tf.keras.layers.Reshape((256, 256, 3))(label_embedding)
    concatenated = tf.keras.layers.Concatenate()([image_input, label_embedding])

    conv2d_1 = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(concatenated)
    pool_1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(conv2d_1)
    conv2d_2 = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(pool_1)
    pool_2 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2)(conv2d_2)

    flatten = tf.keras.layers.Flatten()(pool_2)
    dense_1 = tf.keras.layers.Dense(512, activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(256, activation='relu')(dense_1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_2)
    model = tf.keras.Model(inputs=[image_input, label_input], outputs=output)
    return model


