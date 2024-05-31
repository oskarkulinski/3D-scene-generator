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
    dense_1 = tf.keras.layers.Dense(128, activation='relu')(concatenate)
    dense_2 = tf.keras.layers.Dense(64 * 64 * 3, activation='relu')(dense_1)
    reshape = tf.keras.layers.Reshape((64, 64, 3))(dense_2)
    upsample_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(reshape)
    conv_1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(upsample_1)
    upsample_2 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv_1)
    conv_2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(upsample_2)

    output = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same")(conv_2)

    model = tf.keras.Model(inputs=[noise_input, label_input], outputs=output)
    return model
