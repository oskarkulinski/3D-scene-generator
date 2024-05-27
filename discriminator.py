import tensorflow as tf

import parameters as params


def build_discriminator():
    image_input = tf.keras.Input(shape=(256, 256, 3), name='image_input')
    label_input = tf.keras.Input(shape=(1,), name='label_input')

    label_embedding = tf.keras.layers.Embedding(params.num_classes, 100)(label_input)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)
    label_embedding = tf.keras.layers.Dense(256 * 256)(label_embedding)
    label_embedding = tf.keras.layers.Reshape((256, 256, 3))(label_embedding)

    model = tf.keras.Sequential([
        tf.keras.layers.Concatenate()([image_input, label_embedding]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model
