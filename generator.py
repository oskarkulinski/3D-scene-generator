import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import parameters as params


def build_generator():
    noise_input = tf.keras.Input(shape=(params.noise_dim,), name='noise_input')
    noise_input = tf.keras.layers.Flatten()(noise_input)
    label_input = tf.keras.Input(shape=(params.num_classes,), name='label_input')
    label_embedding = tf.keras.layers.Embedding(params.num_classes, 8)(label_input)
    label_embedding = tf.keras.layers.Dense(100, activation='relu')(label_embedding)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)

    concatenate = tf.keras.layers.Concatenate()([noise_input, label_embedding])
    dense_1 = tf.keras.layers.Dense(256, activation='relu')(concatenate)
    dense_2 = tf.keras.layers.Dense(512, activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(256 * 256 * 3, activation='tanh')(dense_2)
    output = tf.keras.layers.Reshape((256, 256, 3))(dense_3)

    model = tf.keras.Model(inputs=[noise_input, label_input], outputs=output)
    return model

