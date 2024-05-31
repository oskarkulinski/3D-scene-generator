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
    dense_2 = tf.keras.layers.Dense(32 * 32 * 3, activation='relu')(concatenate)
    normal = tf.keras.layers.BatchNormalization(momentum=0.8)(dense_2)
    reshape = tf.keras.layers.Reshape((32, 32, 3))(normal)
    conv_1 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(reshape)
    upsample_2 = tf.keras.layers.UpSampling2D(size=(4, 4))(conv_1)
    #conv_2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(upsample_2)

    output = tf.keras.layers.Conv2D(3, kernel_size=3, padding="same")(upsample_2)

    model = tf.keras.Model(inputs=[noise_input, label_input], outputs=output)
    return model
