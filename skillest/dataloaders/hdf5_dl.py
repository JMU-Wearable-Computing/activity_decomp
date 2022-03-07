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
from skillest.dataloaders.transformations import sample_windows, pad_to_shape, force_to_shape
from random import randrange
import math


class H5Dataset(torch.utils.data.IterableDataset):
    def __init__(self, h5_path, transforms, num_iterations=512 * 8):
        self.transforms = transforms
        self.h5_path = h5_path
        self.h5_file = None
        self.num_iterations = num_iterations
        self.current_iteration = 0

    def get_dataset_keys(self, f):
        keys = []
        f.visit(lambda key: keys.append(key) if isinstance(
            f[key], h5py.Dataset) else None)
        return keys

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self.local_iterations = self.num_iterations
        else:
            self.local_iterations = int(math.ceil(self.num_iterations / 
                float(worker_info.num_workers)))
        return self

    def __next__(self):
        if self.current_iteration >= self.local_iterations:
            raise StopIteration
        
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path)
            self.dataset_keys = self.get_dataset_keys(self.h5_file)

        random_idx = randrange(0, len(self.dataset_keys))
        dataset = self.h5_file[self.dataset_keys[random_idx]][:]

        if self.transforms:
            for transform in self.transforms:
                dataset = transform(dataset)

        data = torch.from_numpy(dataset)

        self.current_iteration += 1
        return data


class H5DataModule(LightningDataModule):
    def __init__(self, hdf5_file: str, transforms: List = None, batch_size=512, num_workers=8, num_iterations=1024):
        super().__init__()
        self.hdf5_file = hdf5_file
        self.transforms = transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_iterations = num_iterations

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            H5Dataset(self.hdf5_file, self.transforms),
            batch_size=self.batch_size,
            num_workers=self.num_workers)


if __name__ == "__main__":
    ds = H5Dataset("test.h5", transforms=[lambda x: force_to_shape(
        x, [111, -1]), lambda x: sample_windows(x, 20)])
    ds = iter(ds)
    for i in range(5):
        print(torch.max(next(ds)))
