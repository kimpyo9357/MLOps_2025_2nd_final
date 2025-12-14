import numpy as np
from datasets import load_dataset

import torch
from torch.utils.data import Dataset

class CIFAR(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.tensor(image)
        label = torch.tensor(label)

        image = image.permute(2, 0, 1)

        return {"image": image,"label": label}


def get_datasets():
    hf_ds = load_dataset("cifar10", cache_dir="../dataset")

    train_images = np.array(hf_ds["train"]["img"], dtype=np.float32) / 255.0
    train_labels = np.array(hf_ds["train"]["label"], dtype=np.int64)

    test_images = np.array(hf_ds["test"]["img"], dtype=np.float32) / 255.0
    test_labels = np.array(hf_ds["test"]["label"], dtype=np.int64)

    train_dataset = CIFAR(train_images, train_labels)
    test_dataset = CIFAR(test_images, test_labels)

    return train_dataset, test_dataset
