from turtle import position
from typing import Iterable, List
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

from pytorch_lightning import LightningDataModule

import h5py
import torch
from skillest.dataloaders import sample_windows, pad_to_shape, force_to_shape
from random import randrange


class H5Dataset(torch.utils.data.IterableDataset):
    def __init__(self, h5_path, transforms):
        self.transforms = transforms
        self.h5_file = h5py.File(h5_path, "r")
        self.dataset_keys = self.get_dataset_keys(self.h5_file)

    def get_dataset_keys(self, f):
        keys = []
        f.visit(lambda key: keys.append(key) if isinstance(
            f[key], h5py.Dataset) else None)
        return keys

    def __iter__(self):
        return self

    def __next__(self):
        random_idx = randrange(0, len(self.dataset_keys))
        dataset = self.h5_file[self.dataset_keys[random_idx]][:]
        if self.transforms:
            for transform in self.transforms:
                dataset = transform(dataset)

        data = torch.from_numpy(dataset)
        return data


class H5DataModule(LightningDataModule):
    def __init__(self, hdf5_file: str, transforms: List = None, batch_size=512, num_workers=8):
        super().__init__()
        self.hdf5_file = hdf5_file
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            H5Dataset(self.hdf5_file, self.transforms),
            batch_size=self.batch_size,
            num_workers=self.num_workers)


if __name__ == "__main__":
    ds = H5Dataset("test.h5", transforms=[lambda x: force_to_shape(
        x, [111, -1]), lambda x: sample_windows(x, 20)])
    for i in range(5):
        print(torch.max(next(ds)))
