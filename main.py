# This is the code used to train and test the models used in "Comparative Study Between Vision Transformer and
# EfficientNet on Marsh Grass Classification". This paper was accepted to FLAIRS 2023 for the main conference track.

import Train, Evaluate

if __name__ == "__main__":
    # current parameters are inline with paper methodology
    num_epochs = 50
    batch_size = 128
    image_size = 33

    model_types = ['ViT', 'efficientnetb0', 'efficientnetb7']
    data_sets = ['data_set_CV1', 'data_set_CV2', 'data_set_CV3', 'data_set_CV4', 'data_set_CV5']

    for mod_type in model_types:
        for index, dataset in enumerate(data_sets):
            data_file = './Data_Dictionaries/'+dataset+".json"
            Train.train(model_type=mod_type, data_file=data_file,
                        save_file="./MarshSpeciesPrediction/models/"+mod_type + "_CV" + str(index + 1),
                        batch_size=batch_size, image_size=image_size, num_epochs=num_epochs)

            Evaluate.predict(model_type=mod_type, data_file=data_file,
                             fold="CV" + str(index + 1), image_size=image_size, batch_size=batch_size)
