import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mydataset import MyDataset
import torchvision.transforms as T
import torch.onnx
import pickle
import random
import yaml
import json
from sklearn.metrics import accuracy_score
import onnx


class ModelCNN(nn.Module):
    def __init__(self):
        super(ModelCNN, self).__init__()
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


def train_one_epoch(model, dataloader, criterion, optimizer, device=torch.device("cuda:0")):
    model.train()
    loss = None
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch)

        loss = criterion(y_pred, y_batch).to(device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss)


def predict(model, dataloader, criterion, device=torch.device("cuda:0")):
    model.eval()

    # PREDICT FOR EVERY ELEMENT OF THE VAL DATALOADER AND RETURN CORRESPONDING LISTS
    losses, predicted_classes, true_classes = [], [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)

            loss = criterion(y_pred, y_batch).to(device)
            losses.append(loss.cpu())
            predicted_classes.extend(torch.argmax(y_pred, dim=-1).cpu())
            true_classes.extend(y_batch.cpu())
    return losses, predicted_classes, true_classes


def train(model, train_dataloader, val_dataloader, criterion, optimizer, device=torch.device("cuda:0"),
          epochs=10, scheduler=None):
    model.to(device)
    for epoch in range(epochs):
        train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        losses, predicted_classes, true_classes = predict(model, val_dataloader, criterion, device)
        scheduler.step(np.mean(losses))

        accuracy = accuracy_score(predicted_classes, true_classes)
        print(f"{epoch}: {accuracy}")


if __name__ == "__main__":
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
    train_transform = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
    val_transform = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])

    train_dataset = MyDataset(X_train, Y_train, transform=train_transform)
    val_dataset = MyDataset(X_test, Y_test, transform=val_transform)

    my_train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
    my_val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params["batch_size"])

    # train model
    my_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    my_model = ModelCNN().to(my_device)

    my_optimizer = torch.optim.Adam(my_model.parameters(), lr=params["optimizer_lr"])
    my_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    my_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(my_optimizer, factor=params["scheduler_factor"],
                                                              patience=params["scheduler_patience"], verbose=True)
    n_epochs = params["n_epochs"]

    train(my_model, my_train_dataloader, my_val_dataloader, my_criterion,
          my_optimizer, my_device, n_epochs, my_scheduler)

    # write metrics
    _, pred_classes, y_classes = predict(my_model, my_train_dataloader, my_criterion, my_device)
    accuracy_train = accuracy_score(pred_classes, y_classes)

    _, pred_classes, y_classes = predict(my_model, my_val_dataloader, my_criterion, my_device)
    accuracy_val = accuracy_score(pred_classes, y_classes)

    with open("metrics.json", "w") as f:
        record = {
            "train": {
                "accuracy": accuracy_train
            },
            "valid": {
                "accuracy": accuracy_val
            }
        }
        json.dump(record, f)

    # write model
    torch.save(my_model.state_dict(), "models/model_pytorch")
