from os.path import join
from typing import Callable, Dict, Optional, Sequence
from pandas.io.parsers import read_table
import torch
import numpy as np
from torch._C import FunctionSchema
from torch.functional import _return_counts
from torch.utils.data import DataLoader, TensorDataset

from pytorch_lightning import LightningDataModule
import pandas as pd

from skillest.dataloaders import (ACTITRCKER_ACTIVITIES,
                                  ACTITRCKER_DIR,
                                  ACTITRCKER_SAMPLE_RATE_PER_SEC,
                                  ACTITRCKER_XML_FILEPATH)
from skillest.utils import get_activity_data_info
from numpy.random import randint
import random

transformation_t = Callable[[np.array], np.array]


class IMUDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 activities: Sequence[str],
                 activities_to_idx: Dict[str, int],
                 sample_rate_per_sec: int,
                 file_prefix: str = ".",
                 batch_size: int = 256,
                 num_seconds: int = 10,
                 num_batches: int = 1000,
                 feature_columns: Sequence[str] = None,
                 transformations: Optional[Sequence[transformation_t]] = None,
                 return_activities: bool = False,
                 return_user: bool = False,
                 **dataloader_kwargs):
        """Creates a new instance of IMUDataModule.

        Args:
            data_dir (str): Directory that the data resides in. Ex: activities/Actitracker
            activities (Sequence[str]): Activities that wanted from the data file.
            activities_to_idx (Dict[str, int]): Mapping of each activity to an int.
            sample_rate_per_sec (int): Sample rate per second that the data represents.
            file_prefix (str, optional): File prefix to add data_dir. Defaults to ".".
            batch_size (int, optional): Batch size for training. Defaults to 256.
            num_seconds (int, optional): Number of seconds per sequence. Total sequence length
                is num_seconds * sample_rate_per_sec. Defaults to 10.
            num_batches (int, optional): Number of batches the DataLoaders will return. Defaults to 1000.
            feature_columns (Sequence[str], optional): Features columns to use from the data. 
                These will automatically be gotten if left unspecified. Defaults to None.
            transformations (Optional[Sequence[Callable[[np.array], np.array]]], optional): List of 
                transformations to apply to the data. Defaults to None.
            return_activities (bool, optional): Should activity labels be returned each batch. Defaults to False.
            return_user (bool, optional): Should user labels be returned each batch. Defaults to False.
        """
        super().__init__()

        self.data_dir = join(file_prefix, data_dir)
        self.activies = activities
        self.activies_to_idx = activities_to_idx
        self.batch_size = batch_size
        self.num_seconds = num_seconds
        self.samples_per_window = self.num_seconds * sample_rate_per_sec
        self.num_batches = num_batches
        self.transformations = transformations
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

        self.save_hyperparameters()

    def get_dataset(self, data: Sequence[Dict[str, np.array]]) -> TensorDataset:
        """Builds dataset from the given reformatted data.

        Args:
            data (Sequence[Dict[str, np.array]]): Data to build from.

        Returns:
            [TensorDataset]: TensorDataset that can iterate over
            windowed, shuffled, and transformed data and labels.
        """
        # Data now conatins windows and optionally user and activity labels.
        data = self.generate_windows(self.train)
        data = self.shuffle(data)
        # Data[0] is the windowed data
        data[0] = self.apply_transformations(data[0])
        # Transform from numpy to pytorch
        data = [torch.tensor(t) for t in data]
        dataset = TensorDataset(*data)
        return dataset

    def sample_windows(self, sample: np.array, count: int) -> np.array:
        """Samples `count` random windows from the given IMU sample. 

        Args:
            sample (np.array): Data to generate windows from.
            count (int): Number of windows to generate from. 

        Returns:
            np.array: Windowed data.
        """
        window_starts = randint(
            sample.shape[0] - self.samples_per_window, size=[count])
        window_ends = window_starts + self.samples_per_window
        windows = np.empty([count, self.samples_per_window, sample.shape[-1]])
        for i in range(count):
            windows[i] = sample[window_starts[i]: window_ends[i]]
        return windows

    def generate_windows(self, data: Sequence[Dict[str, np.array]]) -> Sequence[np.array]:
        """ Generate random windows from user activity data.
        Random numbers of windows are obtained from each user and activity.

        Args:
            data (Sequence[Dict[str, np.array]]): Data to generate windos from.

        Returns:
            [Sequence[np.array]]: windows, user labels (optional), activity labels (optional)
        """
        n_users = len(data)
        channels = len(self.feature_columns)
        total_samples = self.batch_size * self.num_batches

        # Set up empty numpy ararys to load data into
        windows = np.zeros([total_samples, self.samples_per_window, channels])
        if self.return_user:
            user_labels = np.zeros(total_samples)
        if self.return_activities:
            activity_labels = np.zeros(total_samples, dtype='<U20')

        # Decide how many times each user should be included
        users, counts = np.unique(np.random.randint(
            n_users, size=total_samples), return_counts=True)
        total_idx = 0
        for i, user in enumerate(data):
            # Get number of activities
            num_acts = len(user)
            # Incase of unlucky randint that misses a user
            count_idx = np.nonzero(users == i)[0][0]
            user_count = counts[count_idx]

            if self.return_user:
                user_labels[total_idx: total_idx + user_count] = i
            # Chose random activities. Not all users have the same activities. act is array of str.
            act, act_counts = np.unique(random.choices(
                list(user.keys()), k=user_count), return_counts=True)
            for a, ac in zip(act, act_counts):
                end_idx = total_idx + ac
                windows[total_idx: end_idx] = self.sample_windows(user[a], ac)
                if self.return_activities:
                    activity_labels[total_idx: end_idx] = a
                total_idx = end_idx
        assert total_idx == windows.shape[0]

        # Build output array
        out = [windows, ]
        if self.return_user:
            out.append(user_labels)
        if self.return_activities:
            acts, inv = np.unique(activity_labels, return_inverse=True)
            activity_labels = np.array(
                [self.activies_to_idx[a] for a in acts])[inv]
            out.append(activity_labels)

        return out

    def apply_transformations(self, windows: np.array) -> np.array:
        """ Apply transformations to the windows. 

        Args:
            windows (np.array): Windows to apply transformations to.
            shape expected [N, seq_len, num_channels]

        Returns:
            np.array: Windows with transformations applied. 
        """
        for t in self.transformations:
            windows = t(windows)
        return windows

    def shuffle(self, data: Sequence[np.array]) -> Sequence[np.array]:
        """Shuffles the indicies of the data.

        Args:
            data (Sequence[np.array]): Data to shuffle. Should include labels if there are any.

        Returns:
            Sequence[np.array]: Shuffled data and labels.
        """
        N = data[0].shape[0]
        shuffle = np.random.permutation(N)
        return [t[shuffle] for t in data]

    def reformat(self, df: pd.DataFrame) -> Sequence[Dict[str, np.array]]:
        """Reformats the dataframe table to a dict with activity keys mapping to 
        imu data nested in a list indexed by the user idx.

        Args:
            df (pd.DataFrame): Dataframe to format data from. 

        Returns:
            Sequence[Dict[str, np.array]]: Reformated data.
        """
        data = []
        for k, v in df.groupby("UserID"):
            activities = {}
            for a in self.activies:
                activity = np.array(
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

            self.train = self.reformat(train_df)
            self.val = self.reformat(val_df)
        elif stage == "validate":
            val_df = pd.read_csv(self.val_path)
            self.val = self.reformat(val_df)
        elif stage == "test":
            test_df = pd.read_csv(self.test_path)
            self.test = self.reformat(test_df)

    def train_dataloader(self):
        return DataLoader(self.get_dataset(self.train),
                          batch_size=self.batch_size,
                          **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.get_dataset(self.val),
                          batch_size=self.batch_size,
                          **self.dataloader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.get_dataset(self.test),
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
