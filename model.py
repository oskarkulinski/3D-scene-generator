import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import parameters as params
from discriminator import build_discriminator
from generator import build_generator


class SceneGenerator:
    def __init__(self):
        self.discriminator = build_discriminator()
        self.generator = build_generator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.discriminator.trainable = False
        noise_input = tf.keras.layers.Input(shape=(params.noise_dim,))
        label_input = tf.keras.layers.Input(shape=(1,))

        generated_image = self.generator([noise_input, label_input])
        validity = self.discriminator([generated_image, label_input])
        self.gan = tf.keras.Model([noise_input, label_input], validity)
        self.gan.compile(loss='binary_crossentropy', optimizer='adam')

    def generate_noise(self, batch_size, noise_dim):
        return np.random.normal(0, 1, (batch_size, noise_dim))

    def generate_labels(self, batch_size, num_classes):
        return np.random.randint(0, num_classes, batch_size).reshape(-1, 1)

    def train(self, train_dataset, epochs, batch_size, sample_interval):
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of real images and labels
            real_images, real_labels = next(iter(train_dataset))

            batch_size = real_images.shape[0]

            # Generate a batch of fake images
            noise = self.generate_noise(batch_size, params.noise_dim)
            fake_labels = self.generate_labels(batch_size, params.num_classes)
            fake_images = self.generator.predict([noise, fake_labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([real_images, real_labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_images, fake_labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Generate a batch of noise and labels
            noise = self.generate_noise(batch_size, params.noise_dim)
            sampled_labels = self.generate_labels(batch_size, params.num_classes)

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.gan.train_on_batch([noise, sampled_labels], valid)

            # Print the progress
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.2f}%] [G loss: {g_loss}]")

            # If at save interval, save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # Function to save generated image samples
    def sample_images(self, epoch):
        noise = self.generate_noise(10, params.noise_dim)
        sampled_labels = np.arange(0, 10).reshape(-1, 1)
        gen_images = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_images = 0.5 * gen_images + 0.5

        fig, axs = plt.subplots(1, 10, figsize=(20, 2))
        for i in range(10):
            axs[i].imshow(gen_images[i, :, :, 0], cmap='gray')
            axs[i].set_title(f"Class {sampled_labels[i]}")
            axs[i].axis('off')
        plt.show()
