import tensorflow as tf
import parameters as params


def build_generator():
    noise_input = tf.keras.Input(shape=(100,), name='noise_input')
    label_input = tf.keras.Input(shape=(1,), name='label_input')

    label_embedding = tf.keras.layers.Embedding(params.num_classes, 100)(label_input)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)

    concatenate = tf.keras.layers.Concatenate()([noise_input, label_embedding])
    dense_1 = tf.keras.layers.Dense(256, activation='relu')(concatenate)
    dense_2 =tf.keras.layers.Dense(512, activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(256 * 256 * 3, activation='tanh')(dense_2)
    output = tf.keras.layers.Reshape((256, 256, 3))(dense_3)

    model = tf.keras.Model(inputs=[noise_input, label_input], outputs=output)
    return model
