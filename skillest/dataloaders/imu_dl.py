from os.path import join
from typing import Dict, Optional, Sequence
from pandas.io.parsers import read_table
import torch
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule
import pandas as pd

from skillest.dataloaders import (ACTITRCKER_ACTIVITIES, 
                                  ACTITRCKER_DIR,
                                  ACTITRCKER_SAMPLE_RATE_PER_SEC,
                                  ACTITRCKER_XML_FILEPATH)
from skillest.utils import get_activity_data_info
import random

class IMUDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 data: Sequence[Dict[str, torch.tensor]], 
                 samples_per_window: int,
                 batch_size: int, 
                 num_batches: int,
                 return_activities: bool = False,
                 return_user: bool = False):
        """Creates a pytorch dataset for IMU data.

        Args:
            data (Sequence[Dict[torch.tensor]]): Nested list containing a sequence for an activity.
            First list contiains users, and the second is the activities. 
            samples_per_window (int): Number of samples per window.
            batch_size (int): Batch size
            num_batches (int): Number of batches this dataset should return. 
        """
        self.data = data

        self.samples_per_window = samples_per_window
        self.len = batch_size * num_batches
        self.return_activities = return_activities
        self.return_user = return_user

        max_samples = 0
        for u in range(len(data)):
            for a in data[u].keys():
                max_samples = max(max_samples, data[u][a].shape[0])
        # Pre compute with largest window then randomly sample later if it is oob
        # This gives a speed up
        self.window_starts = torch.randint(max_samples - self.samples_per_window, size=[self.len])
    
    def __len__(self):
        return self.len
    def sample_window(self, data: torch.tensor, idx: int) -> torch.tensor:
        """Samples a window from an activity sequence.

        Args:
            data (torch.tensor): Sequence to sample.
            idx (int): Used to get precomputed window start idx.

        Returns:
            [torch.tensor]: Random window from the data. 
        """
        window_start = self.window_starts[idx]
        window_end = window_start + self.samples_per_window
        if data.shape[0] < window_end:
            window_start = random.randrange(data.shape[0] - self.samples_per_window)
            window_end = window_start + self.samples_per_window
        return data[window_start: window_end]

    def __getitem__(self, idx):
        user = random.randrange(len(self.data))
        activity = random.choice(list(self.data[user].keys()))
        window = self.sample_window(self.data[user][activity], idx)

        sample = {"window": window}
        if self.return_activities:
            sample["activity"] = activity
        if self.return_user:
            sample["user"] = user
        return sample


class IMUDataModule(LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 activities: Sequence[str],
                 sample_rate_per_sec: int,
                 file_prefix: str = ".", 
                 batch_size: int = 256,
                 num_seconds: int = 10,
                 num_batches: int = 1000,
                 feature_columns: Sequence[str] = None,
                 return_activities: bool = False,
                 return_user: bool = False,
                 **dataloader_kwargs):
        super().__init__()

        self.data_dir = join(file_prefix, data_dir)
        self.activies = activities
        self.batch_size = batch_size
        self.num_seconds =  num_seconds
        self.samples_per_window = self.num_seconds * sample_rate_per_sec
        self.num_batches = num_batches
        self.return_activities = return_activities
        self.return_user = return_user
        self.dataloader_kwargs = dataloader_kwargs

        self.train_path = join(self.data_dir, "train.csv")
        self.val_path = join(self.data_dir, "val.csv")
        self.test_path = join(self.data_dir, "test.csv")

        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            _, self.feature_columns = get_activity_data_info(self.data_dir)

        # self.save_hyperparameters()

    def separate(self, df: pd.DataFrame):
        data = []
        for k, v in df.groupby("UserID"):
            activities = {}
            for a in self.activies:
                activity = torch.tensor(
                    v[v["Activity"] == a][self.feature_columns]
                    .values
                )
                if activity.shape[0] > 0:
                    activities[a] = activity
            data.append(activities)
        return data

    def setup(self, stage: Optional[str] = None):

        if stage == "fit":
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path)

            self.train = self.separate(train_df)
            self.val = self.separate(val_df)
        elif stage == "validate":
            val_df = pd.read_csv(self.val_path)
            self.val = self.separate(val_df)
        elif stage == "test":
            test_df = pd.read_csv(self.test_path)
            self.test = self.separate(test_df)


    def train_dataloader(self):
        dataset = IMUDataset(self.train, 
                             self.samples_per_window,
                             batch_size=self.batch_size,
                             num_batches=self.num_batches,
                             return_activities=self.return_activities,
                             return_user=self.return_user)
        return DataLoader(dataset, 
                          batch_size=self.batch_size,
                          **self.dataloader_kwargs)

    def val_dataloader(self):
        dataset = IMUDataset(self.val, 
                             self.samples_per_window,
                             batch_size=self.batch_size,
                             num_batches=self.num_batches,
                             return_activities=self.return_activities,
                             return_user=self.return_user)
        return DataLoader(dataset, 
                          batch_size=self.batch_size,
                          **self.dataloader_kwargs)

    def test_dataloader(self):
        dataset = IMUDataset(self.test, 
                             self.samples_per_window,
                             batch_size=self.batch_size,
                             num_batches=self.num_batches,
                             return_activities=self.return_activities,
                             return_user=self.return_user)
        return DataLoader(dataset, 
                          batch_size=self.batch_size,
                          **self.dataloader_kwargs)

    def teardown(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train = None
            self.val = None
        elif stage == "validate":
            self.val = None
        elif stage == "test":
            self.test = None
