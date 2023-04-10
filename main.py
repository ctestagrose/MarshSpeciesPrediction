import os.path

import Train, Evaluate

num_epochs = 30
batch_size = 16
image_size = 33

model_types = ['ViT', 'efficientnetb0', 'efficientnetb7']

if os.path.exists("./models") == False:
    os.makedirs("./models")

for index, mod_type in enumerate(model_types):
    Train.train(model_type=mod_type, data_file="./Data_Dictionaries/data_set_CV" + str(index + 1) + ".json",
          save_file="./models/"+mod_type + "_CV" + str(index + 1),
          batch_size=batch_size, image_size=image_size, num_epochs=num_epochs)