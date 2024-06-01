import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import parameters as params
from discriminator import build_discriminator
from generator import build_generator
from model import SceneGenerator
from data_loader import DataLoader
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train, test = DataLoader.get_train_test()

model = SceneGenerator()

model.train(train, params.epochs, params.batch_size, params.sample_interval, 50)

model.sample_images(25)

model.save_models("saved_models", params.epochs)
