import tensorflow as tf

import parameters as params


def build_discriminator():
    image_input = tf.keras.Input(shape=(256, 256, 3), name='image_input')
    label_input = tf.keras.Input(shape=(1,), name='label_input')

    # output dim set as square root of number of classes
    label_embedding = tf.keras.layers.Embedding(params.num_classes, 1)(label_input)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)
    label_embedding = tf.keras.layers.Dense(256 * 256 * 3, activation="relu")(label_embedding)
    label_embedding = tf.keras.layers.Reshape((256, 256, 3))(label_embedding)
    concatenated = tf.keras.layers.Concatenate()([image_input, label_embedding])

    flatten = tf.keras.layers.Flatten()(concatenated)
    dense_1 = tf.keras.layers.Dense(1024, activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(256, activation='relu')(dense_1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dense_2)
    model = tf.keras.Model(inputs=[image_input, label_input], outputs=output)
    return model
