import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os

import parameters as params


class DataLoader:
    data_dir_path = "./selected_data"

    @staticmethod
    def _split_dataset(images, labels, class_names, test_size=0.2):

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, stratify=labels, random_state=42)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        return train_dataset, test_dataset

    @staticmethod
    def _load_dataset():
        print("Loading dataset from disk...")
        image_list = []
        label_list = []
        class_names = []

        for class_name in sorted(os.listdir(DataLoader.data_dir_path))[:params.num_classes]:
            class_path = os.path.join(DataLoader.data_dir_path, class_name)
            if os.path.isdir(class_path):
                class_names.append(class_name)

                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    try:
                        # Load and preprocess the image
                        image = tf.keras.utils.load_img(image_path, target_size=(params.image_height, params.image_width),
                                                        keep_aspect_ratio=True)
                        image_array = tf.keras.utils.img_to_array(image)
                        image_array = (image_array / 127.5) - 1.0  # Normalize to [-1, 1]
                        image_list.append(image_array)

                        label_list.append(class_names.index(class_name))
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

        # Convert lists to numpy arrays
        images = np.array(image_list)
        labels = np.array(label_list)

        # One-hot encode labels
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))
        DataLoader.cache_dataset(images, labels, class_names)
        return images, labels, class_names

    @staticmethod
    def get_train_test():
        images, labels, class_names = DataLoader.load_cached_data()
        train, test = DataLoader._split_dataset(images, labels, class_names)
        train = train.shuffle(buffer_size=train.cardinality())
        train = train.batch(params.batch_size).prefetch(5)
        print("Loaded data successfully")
        return train, test

    @staticmethod
    def load_cached_data(cache_file='dataset_cache.npz'):
        print("Attempting to load cached data...")
        if os.path.exists(cache_file):
            print("Loading cached data...")
            data = np.load(cache_file)
            images = data['images']
            labels = data['labels']
            class_names = data['class_names']
            return images, labels, class_names
        else:
            print("Failed to load cached data, loading from disk...")
            return DataLoader._load_dataset()

    @staticmethod
    async def cache_dataset(images, labels, class_names, cache_file='dataset_cache.npz'):
        print("Caching data...")
        np.savez_compressed(cache_file, images=images, labels=labels, class_names=class_names)
