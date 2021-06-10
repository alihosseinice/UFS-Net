import pandas as pd
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from pathlib import Path
from pandas import DataFrame
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
import torch.optim as optim
import os
import sys
import time
import copy
from UFS_Net_architecture import UFSNet

batch_size = 64

# Define 8 class labels
Labels = {
    'Normal': [0, 0, 0],
    'Flame': [1, 0, 0],
    'WhiteSmoke': [0, 1, 0],
    'BlackSmoke': [0, 0, 1],
    'Flame_BlackSmoke': [1, 0, 1],
    'Flame_WhiteSmoke': [1, 1, 0],
    'WhiteSmoke_BlackSmoke': [0, 1, 1],
    'Flame_WhiteSmoke_BlackSmoke': [1, 1, 1],
}

classLabels = ['Flame', 'WhiteSmoke', 'BlackSmoke']

# Create Groundtruth for Dataset Images
DataBase = []
for folder in list(Labels.keys()):  # Read all iamges from 8 floder and Create CSV File
    path = 'Dataset/' + folder + '/'
    for filename in os.listdir(path):
        l = []
        l.append(folder + '/' + filename)
        for item in Labels[folder]:
            l.append(item)
        DataBase.append(l)

df = DataFrame(DataBase, columns=['Filename', 'Flame', 'WhiteSmoke', 'BlackSmoke'])
df.to_csv('labels.csv', index=False)


# Create Data pipeline
class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transforms = transforms

    def __getitem__(self, idx):
        d = self.df.iloc[idx]
        image = Image.open(self.img_dir / d.Filename).convert("RGB")
        label = torch.tensor(d[1:].tolist(), dtype=torch.float32)

        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.df)


# Preprocessing Operations
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.4486, 0.4339, 0.4241], [0.2409, 0.2420, 0.2494])
                                ])

# Load Dataset And Create Pytorch Data Loader
dataset = MyDataset("labels.csv", Path("Dataset/"), transform)
valid_no = int(len(dataset) * 0.20)
trainset, valset = random_split(dataset, [len(dataset) - valid_no, valid_no])
print(f"trainset len {len(trainset)} \nvalset len {len(valset)}")
dataloader = {"train": DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=0),
              "val": DataLoader(valset, shuffle=True, batch_size=batch_size, num_workers=0)}

# Model Definition
model = UFSNet(num_classes=3, transform_input=transform)

# Detect devise : GPU or CPU ?   ---> Load model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define Optimizer and Criterion
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
sgdr_partial = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)


# Training Function
def train(model, data_loader, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_lose = 1000.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == "train":  # put the model in training mode
                model.train()
            else:  # put the model in validation mode
                model.eval()

            # keep track of training and validation loss
            running_loss = 0.0
            running_corrects = 0.0

            steps = len(dataloader[phase].dataset) // dataloader[phase].batch_size
            i = 0

            for data, target in data_loader[phase]:
                # load the data and target to respective device
                data, target = data.to(device), target.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    # feed the input
                    output = model(data)
                    output = output.to(device)

                    # calculate the loss
                    loss = criterion(output, target)
                    preds = torch.sigmoid(output).data > 0.5
                    preds = preds.to(torch.float32)

                    if phase == "train":
                        # backward pass: compute gradient of the loss with respect to model parameters
                        loss.backward()
                        # update the model parameters
                        optimizer.step()
                        # zero the grad to stop it from accumulating
                        optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(torch.all(preds == target, axis=1))

                sys.stdout.flush()
                sys.stdout.write("\r  Step %d/%d |  " % (i, steps))
                i += 1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(data_loader[phase].dataset)
            epoch_acc = running_corrects / len(data_loader[phase].dataset)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),
                           "./models/{}-epoch-{}-acc-{:.5f}.pth".format('UFS-Net', epoch + 1, best_acc))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


map_location = torch.device('cuda')
model.eval()

Trained_UFSNet = train(model, dataloader, criterion, optimizer, sgdr_partial, num_epochs=20)
