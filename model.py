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


class SceneGenerator:
    def __init__(self):
        self.discriminator = build_discriminator()
        self.generator = build_generator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.discriminator.trainable = False
        noise_input = tf.keras.layers.Input(shape=(params.noise_dim,))
        label_input = tf.keras.layers.Input(shape=(params.num_classes,))

        generated_image = self.generator([noise_input, label_input])
        validity = self.discriminator([generated_image, label_input])
        self.gan = tf.keras.Model([noise_input, label_input], validity)
        self.gan.compile(loss='binary_crossentropy', optimizer='adam')

    def generate_noise(self, batch_size, noise_dim):
        return np.random.normal(0, 1, (batch_size, noise_dim))

    def generate_labels(self, batch_size, num_classes):
        labels = np.random.randint(0, num_classes, batch_size)
        one_hot_labels = np.zeros((batch_size, num_classes))
        one_hot_labels[np.arange(batch_size), labels] = 1

        return one_hot_labels

    def train(self, train_dataset, epochs, batch_size, sample_interval, save_interval=10):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        folder_name = os.path.join("saved_models", current_datetime)
        os.makedirs(folder_name, exist_ok=True)

        d_loss = 0
        g_loss = 0

        for epoch in range(epochs):
            for real_images, real_labels in train_dataset:
                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.discriminator.trainable = True

                batch_size = real_images.shape[0]
                if batch_size != valid.shape[0]:
                    valid = np.ones((batch_size, 1))
                    fake = np.zeros((batch_size, 1))

                # Generate a batch of fake images
                noise = self.generate_noise(batch_size, params.noise_dim)
                fake_labels = self.generate_labels(batch_size, params.num_classes)
                fake_images = self.generator.predict([noise, fake_labels])
                #print(fake_labels)

                # Train the discriminator
                d_loss_real = self.discriminator.train_on_batch([real_images, real_labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_images, fake_labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Train Generator
                # ---------------------

                self.discriminator.trainable = False

                # Generate a batch of noise and labels
                noise = self.generate_noise(batch_size, params.noise_dim)
                sampled_labels = self.generate_labels(batch_size, params.num_classes)

                # Train the generator (to have the discriminator label samples as valid)
                g_loss = self.gan.train_on_batch([noise, sampled_labels], valid)

            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss}]")

            # If at save interval, save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            if epoch != 0 and epoch % save_interval  == 0:
                sub_folder_name = os.path.join(folder_name, f"epoch_{epoch}")
                os.makedirs(sub_folder_name, exist_ok=True)
                self.save_models(sub_folder_name, epoch)

    async def save_models(self, folder_name, epoch):
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
        #gen_images = 0.5 * gen_images + 0.5

        fig, axs = plt.subplots(1, 5, figsize=(5, 2))
        for i in range(params.num_classes):
            axs[i].imshow(gen_images[i])
            axs[i].set_title(f"Class {sampled_labels[i][sampled_labels[i] == 1]}")
            axs[i].axis('off')
        plt.show()
