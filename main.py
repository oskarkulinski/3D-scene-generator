import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import parameters as params
from discriminator import build_discriminator
from generator import build_generator
from model import SceneGenerator
from data_loader import DataLoader

data_loader = DataLoader()

train, test = data_loader.get_train_test()

model = SceneGenerator()

model.train(train, params.epochs, params.batch_size, params.sample_interval)