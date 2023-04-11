import json
import os
import random
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from monai import transforms
import cv2
from efficientnet_pytorch import EfficientNet
import torch
import torchvision.transforms as T
from monai.data import CacheDataset
from monai.transforms import EnsureType, Activations, LoadImaged, AsChannelFirstD, NormalizeIntensityd, EnsureTyped
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_vit import ViT
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from torchvision.transforms import Compose


class dataloader(Dataset):
    def __init__(self, dict, transforms):
        self.dict = dict
        self.transforms = transforms

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):
        image = cv2.imread(self.dict[index]['image'])
        image = self.transforms(image)
        label = self.dict[index]['label']
        label = torch.FloatTensor(label)
        return image, label

def make_Saliency_Maps(model_type, data_file, fold, image_size, batch_size,):

    test_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(int(image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    if model_type == 'efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6)
        model.load_state_dict(torch.load("MarshSpeciesPrediction/models/" + model_type + "_" + fold + ".pth"))

    if model_type == 'efficientnetb7':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=6)
        model.load_state_dict(torch.load("MarshSpeciesPrediction/models/" + model_type + "_" + fold + ".pth"))

    if model_type == 'ViT':
        model = ViT("B_16", pretrained=True, image_size=33, num_classes=6)
        model.load_state_dict(torch.load("MarshSpeciesPrediction/models/" + model_type + "_" + fold + ".pth"))

    data_file = open(data_file, 'r')
    data_set = json.load(data_file)
    test_set = data_set['Test']
    random.Random(43).shuffle(test_set)
    test_set = test_set[:1]

    image = test_set[0]
    print(image)
    image = test_set[0]
    print(image)

    num_classes = 6

    test_ds = dataloader(test_set, transforms=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=int(batch_size), num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    count = 0
    model.eval()
    for _test in test_loader:
        inputs = _test['image'].to(device)
        inputs.requires_grad_()
        output = model(inputs)
        output_idx = output.argmax()
        output_max = output[0, output_idx]
        output_max.retain_grad()
        output_max.backward()
        saliency, _ = torch.max(inputs.grad.data.abs(), dim=1)
        print(saliency)
        saliency = saliency.reshape(33, 33)
        image = inputs.reshape(-1, 33, 33)
        plt.imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
        plt.imshow(saliency.cpu(), cmap='hot', alpha=0.6)
        plt.title(model_type+' Saliency Map')
        plt.savefig(model_type+"_Saliency_Map.png")
        count += 1


def create_ROC(y_test, y_score, y_preds, num_classes, model_type, fold):
    y_test = np.array([t.ravel() for t in y_test])

    cycol = cycle('rgbymc')
    cylin = cycle('-:.')
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = ['a', 'b', 'm', 's', 'j', 'p']

    for i in range(num_classes):
        if i % 6 == 0:
            lin = next(cylin)
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=next(cycol), ls=lin, lw=lw,
                 label='AUC: {0:0.2f} - {1}'.format(roc_auc[i], labels[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    _fn = 'MarshSpeciesPrediction/AUC_ROC_Curves/' + model_type + '_' + fold + '.png'
    plt.savefig(_fn)
    print('Area under the ROC curve is, ', roc_auc)
    plt.clf()


def predict(model_type, data_file, fold, image_size, batch_size,):

    test_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(int(image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    if model_type == 'efficientnetb0':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6)
        model.load_state_dict(torch.load("MarshSpeciesPrediction/models/" + model_type + "_" + fold + ".pth"))

    if model_type == 'efficientnetb7':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=6)
        model.load_state_dict(torch.load("MarshSpeciesPrediction/models/" + model_type + "_" + fold + ".pth"))

    if model_type == 'ViT':
        model = ViT("B_16", pretrained=True, image_size=33, num_classes=6)
        model.load_state_dict(torch.load("MarshSpeciesPrediction/models/" + model_type + "_" + fold + ".pth"))

    data = open(data_file, 'r')
    data_set = json.load(data)
    test_set = data_set['Test']

    num_classes = 6

    test_ds = dataloader(test_set, transforms=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    Ground_Truth = []
    y_preds = np.zeros([len(test_set), num_classes])
    y_predicted = []
    y_true = []
    y_test = []

    labels = ['a', 'b', 'm', 's', 'j', 'p']

    model.eval()

    count = 0
    for item in test_set:
        Ground_Truth.append(np.argmax(item['label']))

    y_pred_trans = nn.Sigmoid()

    with torch.no_grad():
        for _test in test_loader:
            inputs = _test[0].to(device)
            y_test.append(np.array(_test[1]))
            gt = _test[1]
            y_true.extend((torch.max(torch.exp(gt), 1)[1]).data.cpu().numpy())
            y = model(inputs)
            predicted = (torch.max(torch.exp(y), 1)[1]).data.cpu().numpy()
            y_predicted.extend(predicted)
            y_pred = y_pred_trans(y).to('cpu')
            y_preds[count, :] = y_pred
            count += 1

    y_test = np.array([t.ravel() for t in y_test])
    y_score = np.array([t.ravel() for t in y_preds])

    with open("./MarshSpeciesPrediction/Confusion_Matrices/"+model_type + "_" +data_file[-8:-5]+"_CM.txt", "w") as f:
        cm = confusion_matrix(y_true, y_predicted)
        for line in cm:
            for item in line:
                f.write(str(item)+" ")
            f.write("\n")
        f.close()

    with open("./MarshSpeciesPrediction/Classification_Reports/"+model_type + "_" +data_file[-8:-5]+"_CR.txt", "w") as f:

        f.write(classification_report(y_true, y_predicted))

        f.close()

    create_ROC(y_test, y_score, y_preds, num_classes, model_type, fold)

def make_sample_saliencies(model_type, image_size):

    test_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(int(image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'efficientnetb0':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6)
        model.load_state_dict(torch.load("MarshSpeciesPrediction/models/" + model_type + "_CV1.pth"))

    if model_type == 'efficientnetb7':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=6)
        model.load_state_dict(torch.load("MarshSpeciesPrediction/models/" + model_type + "_CV1.pth"))

    if model_type == 'ViT':
        model = ViT("B_16", pretrained=True, image_size=33, num_classes=6)
        model.load_state_dict(torch.load("MarshSpeciesPrediction/models/" + model_type + "_CV1.pth"))
    model.eval()
    model.to(device)

    set = {"Test":[]}

    for image in os.listdir("./Sample_Images"):
        temp = {"image": os.path.join("./Sample_Images"+"/"+image), "label": [1,0,0,0,0,0]}
        set["Test"].append(temp)

    test_set = set["Test"]

    test_ds = dataloader(test_set, transforms=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, num_workers=8)

    counter = 0
    for _test in test_loader:
        inputs = _test['image'].to(device)
        inputs.requires_grad_()
        output = model(inputs)
        output_idx = output.argmax()
        output_max = output[0, output_idx]
        output_max.retain_grad()
        output_max.backward()
        saliency, _ = torch.max(inputs.grad.data.abs(), dim=1)
        saliency = saliency.reshape(33, 33)
        image = inputs.reshape(-1, 33, 33)
        plt.imshow(image.cpu().detach().numpy().transpose(1, 2, 0))
        plt.imshow(saliency.cpu(), cmap='hot', alpha=0.6)
        plt.title(model_type + ' Saliency Map')
        plt.savefig(model_type+"_Saliency_Map_"+str(counter)+".png")
        counter+=1

if __name__ == "__main__":

    model_types = ['ViT', 'efficientnetb0', 'efficientnetb7']

    for index, mod_type in enumerate(model_types):
        predict(model_type=mod_type, data_file="data_set_CV" + str(index + 1) + ".json", fold="CV"+str(index+1))

    make_sample_saliencies(model_type="ViT")
    make_sample_saliencies(model_type="efficientnet")
    make_sample_saliencies(model_type="efficientnetb7")

