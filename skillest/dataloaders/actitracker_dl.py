from os.path import join
from typing import Optional, Sequence
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

class ActitrackerDataset(torch.utils.data.Dataset):

    def __init__(self, data, 
                 samples_per_window: int,
                 batch_size: int, 
                 num_batches: int):
        self.data = data

        self.samples_per_window = samples_per_window
        self.len = batch_size * num_batches

        max_samples = 0
        for u in range(len(data)):
            for a in range(len(data[u])):
                max_samples = max(max_samples, data[u][a].shape[0])
        # Pre compute with largest window then randomly sample later if it is oob
        # This gives a speed up
        self.window_starts = torch.randint(max_samples - self.samples_per_window, size=[self.len])
    
    def __len__(self):
        return self.len

    def sample_window(self, data, idx):
        window_start = self.window_starts[idx]
        window_end = window_start + self.samples_per_window
        if data.shape[0] < window_end:
            window_start = random.randrange(data.shape[0] - self.samples_per_window)
            window_end = window_start + self.samples_per_window
        return data[window_start: window_end]

    def __getitem__(self, idx):
        user = random.randrange(len(self.data))
        activity = random.randrange(len(self.data[user]))
        window = self.sample_window(self.data[user][activity], idx)
        return window


class ActitrackerDL(LightningDataModule):

    def __init__(self,
                 file_prefix: str = ".", 
                 batch_size: int = 256,
                 num_seconds: int = 10,
                 num_batches: int = 1000,
                 **dataloader_kwargs):
        super().__init__()

        self.data_dir = join(file_prefix, ACTITRCKER_DIR)
        self.xml_filepath = join(file_prefix, ACTITRCKER_XML_FILEPATH)

        self.train_path = join(self.data_dir, "train.csv")
        self.val_path = join(self.data_dir, "val.csv")
        self.test_path = join(self.data_dir, "test.csv")

        self.batch_size = batch_size
        self.num_seconds =  num_seconds
        self.samples_per_window = self.num_seconds * ACTITRCKER_SAMPLE_RATE_PER_SEC
        self.num_batches = num_batches
        self.dataloader_kwargs = dataloader_kwargs

        _, self.feature_columns = get_activity_data_info(ACTITRCKER_DIR)

        # self.save_hyperparameters()

    def separate(self, df: pd.DataFrame):
        data = []
        for k, v in df.groupby("UserID"):
            activities = []
            for a in ACTITRCKER_ACTIVITIES:
                activity = torch.tensor(
                    v[v["Activity"] == a][self.feature_columns]
                    .values
                )
                if activity.shape[0] > 0:
                    activities.append(activity)
            data.append(activities)
        return data

    def setup(self, stage: Optional[str] = None):

        if stage == "fit":
            train_df = pd.read_csv(self.train_path)
            val_df = pd.read_csv(self.val_path)

            self.train = self.separate(train_df)
            self.val = self.separate(val_df)

    def train_dataloader(self):
        dataset = ActitrackerDataset(self.train, 
                                     self.samples_per_window,
                                     batch_size=self.batch_size,
                                     num_batches=self.num_batches)
        return DataLoader(dataset, 
                          batch_size=self.batch_size,
                          **self.dataloader_kwargs)

    def teardown(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train = None
            self.val = None

            
if __name__ == "__main__":
    dl = ActitrackerDL(num_batches=1000, num_workers=8)
    dl.setup("fit")
    d = dl.train_dataloader()
    from datetime import datetime
    start = datetime.now()
    for idx, sample in enumerate(d):
        # print(f"{idx}: sample")
        pass
    end = datetime.now()
    dl.teardown("fit")
    print(f"Duration: {end - start}")

