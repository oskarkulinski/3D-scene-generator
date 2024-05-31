import os
from model import SceneGenerator
from data_loader import DataLoader
import parameters as params
import numpy as np

print("Do you wish to train the model or load an existing one?")
print("1 = Train the model, 2 = Load an existing model")
train_choice = input()
model = SceneGenerator()
if train_choice == "1":
    train, test = DataLoader.get_train_test()

    model.train(train, params.epochs, params.batch_size, params.sample_interval)

else:
    model.load_models("main_model")


print("Choose your class")
classes = []
i = 0
for cls in os.listdir("./selected_data"):
    classes.append(cls)
    print("{cls} --> {i}".format(cls=cls, i=i), end="")
while True:
    number = input()
    if number in range(0, len(classes)):
        noise = np.random.normal(0, 1, 5)
        labels = np.zeros(shape=(params.num_classes, 5))
        for i in range(5):
            labels[number][i] = 1
        model.generator.predict([noise, labels])()
    else:
        break
