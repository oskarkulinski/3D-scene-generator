import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os

import parameters as params


class DataLoader:
    data_dir_path = "./selected_data"

    @staticmethod
    def _load_dataset():
        print("Loading dataset from disk...")
        image_list = []

        for class_name in sorted(os.listdir(DataLoader.data_dir_path)):
            class_path = os.path.join(DataLoader.data_dir_path, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    try:
                        # Load and preprocess the image
                        image = tf.keras.utils.load_img(image_path, target_size=(params.image_height, params.image_width))
                        image_array = tf.keras.utils.img_to_array(image)
                        image_array = (image_array / 127.5) - 1 # Normalize to [-1, 1]
                        image_list.append(image_array)

                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

        images = np.array(image_list)

        DataLoader.cache_dataset(images)
        return images

    @staticmethod
    def get_data():
        images = DataLoader.load_cached_data()
        train = tf.data.Dataset.from_tensor_slices(images)
        train = train.shuffle(buffer_size=1000)
        train = train.batch(params.batch_size).prefetch(5)
        print("Loaded data successfully")
        return train

    @staticmethod
    def load_cached_data(cache_file='dataset_cache.npz'):
        print("Attempting to load cached data...")
        if os.path.exists(cache_file):
            print("Loading cached data...")
            data = np.load(cache_file)
            images = data['images']
            return images
        else:
            print("Failed to load cached data, loading from disk...")
            return DataLoader._load_dataset()

    @staticmethod
    def cache_dataset(images, cache_file='dataset_cache.npz'):
        print("Caching data...")
        np.savez_compressed(cache_file, images=images)

