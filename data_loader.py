import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os

import parameters as params


class DataLoader:
    data_dir_path = "./data/indoorCVPR_09/Images/"

    @staticmethod
    def _split_dataset(images, labels, class_names, test_size=0.2):

        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, stratify=labels, random_state=42)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        return train_dataset, test_dataset

    @staticmethod
    def _load_dataset():
        image_list = []
        label_list = []
        class_names = []

        for class_name in sorted(os.listdir(DataLoader.data_dir_path)):
            class_path = os.path.join(DataLoader.data_dir_path, class_name)
            if os.path.isdir(class_path):
                class_names.append(class_name)

                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    try:
                        # Load and preprocess the image
                        image = tf.keras.utils.load_img(image_path, target_size=params.image_size,
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

        return images, labels, class_names

    @staticmethod
    def get_train_test():
        images, labels, class_names = DataLoader._load_dataset()
        train, test = DataLoader._split_dataset(images, labels, class_names)
        train = train.shuffle(buffer_size=train.cardinality())
        train = train.batch(params.batch_size).prefetch(5)
        return train, test
