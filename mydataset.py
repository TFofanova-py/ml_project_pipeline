import torch
import numpy as np


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
        return img, label

    def __len__(self):
        return len(self.images)