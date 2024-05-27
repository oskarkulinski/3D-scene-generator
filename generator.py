import tensorflow as tf
import parameters as params


def build_generator():
    noise_input = tf.keras.Input(shape=(100,), name='noise_input')
    label_input = tf.keras.Input(shape=(1,), name='label_input')

    label_embedding = tf.keras.layers.Embedding(params.num_classes, 100)(label_input)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)

    model = tf.keras.Sequential([
        tf.keras.layers.Concatenate()([noise_input, label_embedding]),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256 * 256 * 3, activation='tanh'),
        tf.keras.layers.Reshape((256, 256, 3))
    ])

    return model
