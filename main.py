import tensorflow as tf
import os
import parameters as params
from model import SceneGenerator
from data_loader import DataLoader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.random.set_seed(44)

train = DataLoader.get_data()

model = SceneGenerator()

model.train(train, params.epochs)

model.sample_images()

model.save_models("saved_models", epoch=params.epochs)
