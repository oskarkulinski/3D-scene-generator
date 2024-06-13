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

save = input("Would you like to save the generated images to disk? [y/n]")
i = 1
while True:
    decision = input("If you want to quit type q, otherwise type g")
    if decision == "q":
        break
    else:
        if save.lower() == "y":
            model.save_generated_images(i)
            i += 1
        else:
            model.sample_images()

