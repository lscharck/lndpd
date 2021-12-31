import os
import torch
import numpy as np
from torch.utils.data import Dataset

class LandingPadH(Dataset):
    def __init__(self, label_file, root_dir, transform=None):
        self.label_file = np.load(label_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = np.load(os.path.join(self.root_dir, "image" + str(idx) +
            ".npy"))
        label = self.label_file[idx]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = (torch.unsqueeze(torch.from_numpy(image).float(), 0))

        return image, torch.from_numpy(label).float()
