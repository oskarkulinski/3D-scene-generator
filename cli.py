from model import SceneGenerator
from data_loader import DataLoader
import parameters as params

print("Do you wish to train the model or load an existing one?")
print("1 = Train the model, 2 = Load an existing model")
train_choice = input()
model = SceneGenerator()
if train_choice == "1":
    train, test = DataLoader.get_data()

    model.train(train, params.epochs)

else:
    model.load_models("main_model")

while True:
    decision = input("If you want to quit type q, otherwise type g")
    if decision == "q":
        break
    else:
        model.sample_images()

