import tensorflow as tf
from sklearn.model_selection import train_test_split
from pathlib import Path

IMAGE_SIZE = (256, 256)


class DataLoader:
    def __init__(self):
        self.data_dir_path = Path("./data/indoorCVPR_09/")

    def get_paths(self):
        image_paths = list(self.data_dir_path.glob('*/*.jpg'))
        labels = [path.parent.name for path in image_paths]
        label_names = sorted(set(labels))
        label_to_index = {name: index for index, name in enumerate(label_names)}
        labels = [label_to_index[label] for label in labels]
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )
        return train_paths, test_paths, train_labels, test_labels

    def load_and_preprocess_image(self, path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = (image / 127.5) - 1.0  # Normalize to [-1, 1]
        return image, label

    def create_dataset(self, image_paths, labels, batch_size=32, shuffle=True):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths))
            dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def get_train_test(self):
        train_paths, test_paths, train_labels, test_labels = self.get_paths()
        train_dataset = self.create_dataset(train_paths, train_labels)
        test_dataset = self.create_dataset(test_paths, test_labels, shuffle=False)
        return train_dataset,test_dataset
