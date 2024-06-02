import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import parameters as params


def build_generator():
    noise_input = tf.keras.Input(shape=(params.noise_dim,), name='noise_input')
    label_input = tf.keras.Input(shape=params.num_classes, name='label_input')

    label_embedding = tf.keras.layers.Embedding(params.num_classes, 5)(label_input)
    label_embedding = tf.keras.layers.Flatten()(label_embedding)

    concatenated_input = tf.keras.layers.Concatenate()([noise_input, label_embedding])

    dense_1 = tf.keras.layers.Dense(128 * 8 * 8, activation='relu')(concatenated_input)
    normal_1 = tf.keras.layers.BatchNormalization(momentum=0.8)(dense_1)
    reshape = tf.keras.layers.Reshape((8, 8, 128))(normal_1)

    upsample_1 = tf.keras.layers.UpSampling2D(size=(2, 2))(reshape)  # 16x16
    conv_1 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, padding='same', activation='relu')(upsample_1)
    drop_1 = tf.keras.layers.Dropout(0.25)(conv_1)
    normal_2 = tf.keras.layers.BatchNormalization(momentum=0.8)(drop_1)

    upsample_2 = tf.keras.layers.UpSampling2D(size=(2, 2))(normal_2)  # 32x32
    conv_2 = tf.keras.layers.Conv2DTranspose(256, kernel_size=3, padding='same', activation='relu')(upsample_2)
    drop_2 = tf.keras.layers.Dropout(0.25)(conv_2)
    normal_3 = tf.keras.layers.BatchNormalization(momentum=0.8)(drop_2)

    upsample_3 = tf.keras.layers.UpSampling2D(size=(2, 2))(normal_3)  # 64x64
    conv_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, padding='same', activation='relu')(upsample_3)
    drop_3 = tf.keras.layers.Dropout(0.25)(conv_3)
    normal_4 = tf.keras.layers.BatchNormalization(momentum=0.8)(drop_3)

    upsample_4 = tf.keras.layers.UpSampling2D(size=(2, 2))(normal_4)  # 128x128
    conv_4 = tf.keras.layers.Conv2DTranspose(128, kernel_size=3, padding='same', activation='relu')(upsample_4)
    drop_4 = tf.keras.layers.Dropout(0.25)(conv_4)
    normal_5 = tf.keras.layers.BatchNormalization(momentum=0.8)(drop_4)

    output = tf.keras.layers.Conv2D(3, kernel_size=3, padding='same', activation='tanh')(normal_5)

    model = tf.keras.Model(inputs=[noise_input, label_input], outputs=output)
    return model
