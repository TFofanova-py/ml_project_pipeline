import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.io import read_image
import torchvision.transforms as T
import pickle
import random
import yaml
from sklearn.metrics import accuracy_score
import os


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X_data, Y_data, transform):
        self.images = X_data
        self.labels = Y_data
        self.transform = transform

    def __getitem__(self, idx):
        img = np.expand_dims(self.images[idx], -1)
        if self.transform is not None:
            img = self.transform(img)

        label = np.argmax(self.labels[idx])
        return (img, label)

    def __len__(self):
        return len(self.images)


class Model_cnn(nn.Module):
    def __init__(self):
        super(Model_cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(29 * 29 * 16, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        out = F.elu(self.conv1(x))
        out = F.max_pool2d(F.elu(self.conv2(out)), kernel_size=2, stride=2)
        out = torch.nn.Flatten(start_dim=1, end_dim=-1)(out)
        out = F.elu(self.fc1(out))
        out = F.softmax(self.fc2(out), dim=-1)
        return out

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_one_epoch(model, train_dataloader, criterion, optimizer, device="cuda:0"):
    model.train()
    for x_batch, y_batch in train_dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)
        
        loss = criterion(y_pred, y_batch).to(device)
                
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)
        


def predict(model, val_dataloder, criterion, device="cuda:0"):
    model.eval()
    
    # PREDICT FOR EVERY ELEMENT OF THE VAL DATALOADER AND RETURN CORRESPONDING LISTS
    losses, predicted_classes, true_classes = [], [], []
    
    with torch.no_grad():
        for x_batch, y_batch in val_dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
    
            loss = criterion(y_pred, y_batch).to(device)
            losses.append(loss.cpu()) 
            predicted_classes.extend(torch.argmax(y_pred, dim=-1).cpu())
            true_classes.extend(y_batch.cpu())
    return losses, predicted_classes, true_classes


def train(model, train_dataloader, val_dataloader, criterion, optimizer, device="cuda:0", n_epochs=10, scheduler=None):
    model.to(device)
    for epoch in range(n_epochs):
        
        train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        losses, pred_classes, true_classes = predict(model, val_dataloader, criterion, device)
        scheduler.step(np.mean(losses))
        
        accuracy = accuracy_score(pred_classes, true_classes)
        print(f"{epoch}: {accuracy}")


set_random_seed(120)       

with open("datasets/x_train.pickle", "rb") as f:
	X_train = pickle.load(f)
with open("datasets/x_test.pickle", "rb") as f:
	X_test = pickle.load(f)
with open("datasets/y_train.pickle", "rb") as f:
	Y_train = pickle.load(f)
with open("datasets/y_test.pickle", "rb") as f:
	Y_test = pickle.load(f)
	
with open("params.yaml", "r") as f:
	params = yaml.safe_load(f)["train_model"]

# make dataloaders  
train_transform = T.Compose([T.ToTensor(), T.Normalize((0.5), (0.5))])
val_transform = T.Compose([T.ToTensor(), T.Normalize((0.5), (0.5))])

train_dataset = MyDataset(X_train, Y_train, transform=train_transform)
val_dataset = MyDataset(X_test, Y_test, transform=val_transform)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params["batch_size"])


# train model
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = Model_cnn().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=params["optimizer_lr"])
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=params["scheduler_factor"], patience=params["scheduler_patience"], verbose=True)
n_epochs = params["n_epochs"]


train(model, train_dataloader, val_dataloader, criterion, optimizer, device, n_epochs, scheduler)

os.makedirs("models", exist_ok=True)

with open("models/model.pickle", "wb") as f:
	pickle.dump(model, f)
