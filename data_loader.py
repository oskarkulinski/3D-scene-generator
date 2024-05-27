import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

import parameters as params


class DataLoader:
    def __init__(self):
        self.data_dir_path = Path("./data/indoorCVPR_09/Images/")

    def _split_dataset(self, dataset, test_size=0.2):
        images, labels = [], []
        for image, label in dataset:
            images.append(image.numpy())
            labels.append(label.numpy())

        X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, stratify=labels, random_state=42)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        return train_dataset, test_dataset

    def _preprocess_image(self, image, label):
        image = tf.image.resize(image, params.image_size)
        image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
        return image, label

    def _load_dataset(self):
        dataset = tf.keras.preprocessing.image_dataset_from_directory(self.data_dir_path)
        dataset = dataset.map(self._preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def get_train_test(self):
        dataset = self._load_dataset()
        train, test = self._split_dataset(dataset)
        train = train.shuffle(buffer_size=train.cardinality())
        train = train.batch(params.batch_size).prefetch(5)
        return train, test
