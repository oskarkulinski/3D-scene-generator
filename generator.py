import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import parameters as params


def build_generator():
    noise_input = tf.keras.Input(shape=(params.noise_dim,), name='noise_input')
    label_input = tf.keras.Input(shape=params.num_classes, name='label_input')
    # num_classes * output_dim + noise_dim = reshape total size
    noise_reshape = tf.keras.layers.Dense(32448, activation='relu')(noise_input)
    noise_reshape = tf.keras.layers.Reshape((8, 8, 507))(noise_reshape)

    label_embedding = tf.keras.layers.Embedding(params.num_classes, 64)(label_input)
    label_embedding = tf.keras.layers.Reshape((8, 8, 5))(label_embedding)

    concatenated_input = tf.keras.layers.Concatenate()([noise_reshape, label_embedding])

    conv_1 = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, strides=2,
                                             padding='same', activation='relu')(concatenated_input)
    normal_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_1)

    conv_2 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2,
                                             padding='same', activation='relu')(normal_2)
    normal_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_2)

    conv_3 = tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2,
                                             padding='same', activation='relu')(normal_3)
    normal_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_3)

    #conv_4 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, strides=2,
     #                                        padding='same', activation='relu')(normal_4)
    #normal_5 = tf.keras.layers.BatchNormalization(momentum=0.8)(conv_4)

    output = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=2,
                                    padding='same', activation='tanh')(normal_4)

    model = tf.keras.Model(inputs=[noise_input, label_input], outputs=output)
    return model
