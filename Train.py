import json
import os
import cv2
import time
from efficientnet_pytorch import EfficientNet
import torch
import torchvision.transforms as T
from PIL import Image
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_vit import ViT


class dataloader(Dataset):
    def __init__(self, dict, transforms):
        self.dict = dict
        self.transforms = transforms

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):
        img_path = self.dict[index]['image']
        image = cv2.imread(img_path)
        image = self.transforms(image)
        label = self.dict[index]['label']
        label = torch.FloatTensor(label)
        return image, label

def train(model_type, data_file, batch_size, image_size, save_file, num_epochs):
    batch_size = batch_size

    train_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(int(image_size)),
        T.RandomHorizontalFlip(p=0.50),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=(0, 180)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    validation_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(int(image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )

    if model_type == 'efficientnetb0':
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=6)
    elif model_type == 'efficientnetb7':
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=6)
    elif model_type == 'ViT':
        model = ViT("B_16", pretrained=True, image_size=int(image_size), num_classes=6)

    data_file = open(data_file, 'r')
    data_set = json.load(data_file)
    train_set = data_set['Train']
    val_set = data_set['Validation']

    train_ds = dataloader(train_set, transforms=train_transforms)
    validation_ds = dataloader(val_set, transforms=validation_transforms)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=int(batch_size), num_workers=8)
    val_loader = torch.utils.data.DataLoader(validation_ds, batch_size=int(batch_size), num_workers=8)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    epochs = num_epochs

    # training loop
    best_acc = -1
    best_val_loss = 1000
    best_metric_epoch = -1
    epoch_loss_values = []
    acc_values = []
    val_interval = 1

    print("Training "+model_type+"...")
    for epoch in range(epochs):
        print("-" * 10)
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].type(torch.float).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss: .4f}")
        if (epoch + 1) % int(val_interval) == 0:
            model.eval()
            num_correct = 0.0
            metric_count = 0
            val_epoch_loss = 0
            val_step = 0
            with torch.no_grad():
                for val_data in val_loader:
                    val_step += 1
                    val_images, val_labels = val_data[0].to(device), val_data[1].type(torch.float).to(device)
                    val_output = model(val_images)
                    value = torch.eq(val_output.argmax(dim=1), val_labels.argmax(dim=1))
                    val_loss = criterion(val_output, val_labels)
                    val_epoch_loss += val_loss.item()
                    metric_count += len(value)
                    num_correct += value.sum().item()
                    val_epoch_len = len(validation_ds) // val_loader.batch_size
                acc = num_correct / metric_count
                current_val_loss = val_epoch_loss/val_epoch_len
                acc_values.append(acc)
                if current_val_loss < best_val_loss:
                    best_metric = acc
                    best_val_loss = val_epoch_loss/val_epoch_len
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), save_file)
                    print('Saved new model')
                print("Current Epoch: {} current Validation loss: {:.4f}"
                      " Best validation loss: {:.4f} with accuracy: {:.4f} at epoch {}".format(epoch + 1, current_val_loss, best_val_loss, best_metric, best_metric_epoch))
    print(f"Training completed, best_metric: {best_metric: .4f}"
          f" at epoch: {best_metric_epoch}")