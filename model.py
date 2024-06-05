import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import datetime

import parameters as params
from discriminator import build_discriminator
from generator import build_generator
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

tf.random.set_seed(42)


class SceneGenerator:
    def __init__(self):
        self.discriminator = build_discriminator()
        self.generator = build_generator()

        self.generator_optimizer = tf.keras.optimizers.Adam(3e-4,0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1.0e-4,0.5)

        noise_input = tf.keras.layers.Input(shape=(params.noise_dim,))
        label_input = tf.keras.layers.Input(shape=(params.num_classes,))

        generated_image = self.generator([noise_input, label_input])
        validity = self.discriminator([generated_image, label_input])
        self.gan = tf.keras.Model([noise_input, label_input], validity)

    def generate_noise(self, batch_size, noise_dim):
        return np.random.normal(0, 1, (batch_size, noise_dim))

    def generate_labels(self, batch_size, num_classes):
        labels = np.random.randint(0, num_classes, batch_size)
        one_hot_labels = np.zeros((batch_size, num_classes))
        one_hot_labels[np.arange(batch_size), labels] = 1

        return one_hot_labels

    def discriminator_loss(self, real_output, fake_output):
        real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
        fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)


    def train(self, train_dataset, epochs, batch_size, sample_interval, save_interval=10):
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = os.path.join("saved_models", current_datetime)
        os.makedirs(folder_name, exist_ok=True)

        d_loss = 0
        g_loss = 0

        for epoch in range(epochs):
            gen_loss_list = []
            disc_loss_list = []

            # training step on one batch
            for real_images, real_labels in train_dataset:

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    batch_size = real_images.shape[0]

                    # Generate a batch of fake images
                    noise = self.generate_noise(batch_size, params.noise_dim)
                    fake_labels = self.generate_labels(batch_size, params.num_classes)
                    #fake_images = self.generator.predict([noise, fake_labels])

                    fake_images = self.generator([noise, fake_labels], training=True)

                    real_output = self.discriminator([real_images, real_labels], training=True)
                    fake_output = self.discriminator([fake_images, fake_labels], training=True)

                    gen_loss = self.generator_loss(fake_output)
                    disc_loss = self.discriminator_loss(real_output, fake_output)

                    gradients_of_generator = gen_tape.gradient(gen_loss,
                                                               self.generator.trainable_variables)
                    gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                                    self.discriminator.trainable_variables)

                    self.generator_optimizer.apply_gradients(zip(
                        gradients_of_generator, self.generator.trainable_variables))
                    self.discriminator_optimizer.apply_gradients(zip(
                        gradients_of_discriminator,
                        self.discriminator.trainable_variables))

            # calculating total loss in the epoch
            gen_loss_list.append(sum(gen_loss.numpy()))
            disc_loss_list.append(sum(disc_loss.numpy()))

            g_loss: float = sum(gen_loss_list) / len(gen_loss_list)
            d_loss: float = sum(disc_loss_list) / len(disc_loss_list)
            print(f"{epoch}: [D loss: {d_loss}, [G loss: {g_loss}]")

            # If at save interval, save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            if epoch != 0 and epoch % save_interval == 0:
                sub_folder_name = os.path.join(folder_name, f"epoch_{epoch}")
                os.makedirs(sub_folder_name, exist_ok=True)
                self.save_models(sub_folder_name, epoch)

    def save_models(self, folder_name, epoch):
        discriminator_path = os.path.join(folder_name, "discriminator.h5")
        generator_path = os.path.join(folder_name, "generator.h5")
        self.discriminator.save(discriminator_path)
        self.generator.save(generator_path)
        print(f"Models saved for epoch {epoch} at {folder_name}")

    def load_models(self, folder_name):
        discriminator_path = os.path.join(folder_name, "discriminator.h5")
        generator_path = os.path.join(folder_name, "generator.h5")
        self.discriminator.load_weights(discriminator_path)
        self.generator.load_weights(generator_path)
        print(f"Models loaded from {folder_name}")

    def sample_images(self, epoch):
        noise = self.generate_noise(params.num_classes, params.noise_dim)
        sampled_labels = self.generate_labels(5, params.num_classes)
        gen_images = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_images = 0.5 * gen_images + 0.5

        fig, axs = plt.subplots(1, 5, figsize=(5, 2))
        for i in range(params.num_classes):
            index = 0
            for j in range(5):
                if sampled_labels[j][i] == 1:
                    index = j
                    break
            axs[i].imshow((gen_images[i] * 255).astype(np.uint8))
            axs[i].set_title(f"{index}")
            axs[i].axis('off')
        plt.show()
