from turtle import position
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

from pytorch_lightning import LightningDataModule

# DO NOT CHANGE RANDOM SEED ONCE IT IS SELECTED
# This would lead to different builds of the dataset
# splits every time
RANDOM_SEED = 7
###############################################
NUM_SUBJECTS = 10
NUM_MOVEMENTS = 10
NUM_EPISODES = 10
CORRECT = 1
INCORRECT = 0


def get_data():
    # If this file has been updated more recently than the data file
    # recompute the data file
    if os.path.exists("UI-PRMD/kinect_data.pkl"):
        this_file_last_mod = os.path.getmtime(
            "skillest/dataloaders/ui_prmd_dl.py")
        data_pkl_last_mod = os.path.getmtime("UI-PRMD/kinect_data.pkl")
        if this_file_last_mod < data_pkl_last_mod:
            return pd.read_pickle("UI-PRMD/kinect_data.pkl")

    def load_data(angle_dir, positions_dir, data_dict, label):

        file_list_angle = [os.path.join(angle_dir, f)
                           for f in os.listdir(angle_dir)]
        file_list_angle.sort()
        file_list_positions = [os.path.join(
            positions_dir, f) for f in os.listdir(positions_dir)]
        file_list_positions.sort()

        for af, pf in zip(file_list_angle, file_list_positions):
            with open(af, "r") as oaf, open(pf, "r") as opf:
                data_dict["movement"].append(
                    int(af.split("/")[-1].split("_")[0][1:]))
                data_dict["subject"].append(
                    int(af.split("/")[-1].split("_")[1][1:]))
                data_dict["episode"].append(
                    int(af.split("/")[-1].split("_")[2][1:]))
                data_dict["angles"].append(pd.read_csv(
                    oaf, sep=',', header=None).values)
                data_dict["positions"].append(pd.read_csv(
                    opf, sep=',', header=None).values)
                data_dict["label"].append(label)

    inc_angle_dir = "UI-PRMD/Incorrect Segmented Movements/Kinect/Angles"
    inc_positions_dir = "UI-PRMD/Incorrect Segmented Movements/Kinect/Positions"
    cor_angle_dir = "UI-PRMD/Segmented Movements/Kinect/Angles"
    cor_positions_dir = "UI-PRMD/Segmented Movements/Kinect/Positions"

    data = {"movement": [], "subject": [], "episode": [],
            "angles": [], "positions": [], "label": []}
    load_data(cor_angle_dir, cor_positions_dir, data, CORRECT)
    load_data(inc_angle_dir, inc_positions_dir, data, INCORRECT)

    max_len = 0
    for a, p in zip(data["angles"], data["positions"]):
        assert a.shape == p.shape, f"{a.shape} != {p.shape}"
        if a.shape[0] > max_len:
            max_len = a.shape[0]

    for i in range(len(data["angles"])):
        angles = data["angles"][i]
        positions = data["positions"][i]
        data["angles"][i] = np.pad(
            angles, [(0, max_len - angles.shape[0]), (0, 0)])
        data["positions"][i] = np.pad(
            positions, [(0, max_len - positions.shape[0]), (0, 0)])

    df = pd.DataFrame(data)
    df.to_pickle("UI-PRMD/kinect_data.pkl")
    return df


class UIPRMDDataloader(LightningDataModule):

    def __init__(self, batch_size: int,
                 num_sub_in_train: int = None,
                 num_sub_in_val: int = None,
                 num_sub_in_test: int = None,
                 num_mov_in_train: int = None,
                 num_mov_in_val: int = None,
                 num_mov_in_test: int = None,
                 num_ep_in_train: int = None,
                 num_ep_in_val: int = None,
                 num_ep_in_test: int = None):
        super().__init__()
        self.data = get_data()
        self.batch_size = batch_size
        self.rng = np.random.RandomState(RANDOM_SEED)

        self.sub_train, self.sub_val, self.sub_test = self.get_choice_sample(
            NUM_SUBJECTS, num_sub_in_train, num_sub_in_val, num_sub_in_test)

        self.mov_train, self.mov_val, self.mov_test = self.get_choice_sample(
            NUM_MOVEMENTS, num_mov_in_train, num_mov_in_val, num_mov_in_test)

        self.ep_train, self.ep_val, self.ep_test = self.get_choice_sample(
            NUM_EPISODES, num_ep_in_train, num_ep_in_val, num_ep_in_test)
        self.save_hyperparameters()

    def get_choice_sample(self, total, train, val, test):
        sampled_train = np.arange(1, total + 1)
        sampled_val = np.arange(1, total + 1)
        sampled_test = np.arange(1, total + 1)
        # use all values unless we want to sample
        if (train is not None) and (val is not None) and (test is not None):
            sampled_train = self.rng.choice(
                sampled_train, train, replace=False)

            # Remove already sampeld values
            sampled_val_choices = np.delete(sampled_val, sampled_train - 1)
            sampled_val = self.rng.choice(
                sampled_val_choices, val, replace=False)

            sampled_test_choices = np.delete(
                sampled_test, np.concatenate([sampled_train, sampled_val]) - 1)
            sampled_test = self.rng.choice(
                sampled_test_choices, test, replace=False)

        return sampled_train, sampled_val, sampled_test

    def sample_data(self, sub, mov, ep):
        expression = (self.data["subject"].isin(sub)
                      & self.data["movement"].isin(mov)
                      & self.data["episode"].isin(ep))
        sampled_data = self.data[expression]
        angles = np.array([a for a in sampled_data["angles"].values])
        positions = np.array([a for a in sampled_data["positions"].values])
        values = np.concatenate([angles, positions], axis=-1)

        N = values.shape[0]
        mask = self.rng.permutation(N)
        values = torch.tensor(values[mask])
        labels = torch.tensor(sampled_data["label"].values[mask])
        movement = torch.tensor(sampled_data["movement"].values[mask])
        subject = torch.tensor(sampled_data["subject"].values[mask])

        return values, labels, movement, subject

    def train_dataloader(self):
        values, labels, movement, subject = self.sample_data(
            self.sub_train, self.mov_train, self.ep_train)
        batch_size = self.batch_size
        if batch_size == -1:
            batch_size = values.shape[0]
        return DataLoader(TensorDataset(values, labels), batch_size=batch_size)

    def val_dataloader(self):
        values, labels, movement, subject = self.sample_data(
            self.sub_val, self.mov_val, self.ep_val)
        batch_size = self.batch_size
        if batch_size == -1:
            batch_size = values.shape[0]
        return DataLoader(TensorDataset(values, labels), batch_size=batch_size)

    def test_dataloader(self):
        values, labels, movement, subject = self.sample_data(
            self.sub_test, self.mov_test, self.ep_test)
        batch_size = self.batch_size
        if batch_size == -1:
            batch_size = values.shape[0]
        return DataLoader(TensorDataset(values, labels), batch_size=batch_size)


class UIPRMDSingleDataloader(UIPRMDDataloader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subject = None
        self.movement = None

    def set_subject(self, subject: int) -> None:
        assert 1 <= subject and subject <= 10, f"Invalid subject number {subject}"
        self.subject = subject

    def set_movement(self, movement: int) -> None:
        assert 1 <= movement and movement <= 10, f"Invalid activity number {movement}"
        self.movement = movement

    def sample_data(self, sub, mov, ep):
        values, labels, movement, subject = super().sample_data(sub, mov, ep)
        mask = torch.ones_like(labels, dtype=torch.bool)
        if self.subject:
            mask &= (subject == self.subject)
        if self.movement:
            mask &= (movement == self.movement)

        return values[mask, ...], labels[mask], movement[mask], subject[mask]

if __name__ == "__main__":

    # dl = UIPRMDDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2)
    # print(dl.hparams)
    # dl.setup("fit")
    # dt = iter(dl.train_dataloader())
    # dv = iter(dl.val_dataloader())
    # from datetime import datetime
    # start = datetime.now()
    # for tbatch, vbatch in zip(dt, dv):
    #     print(f"{tbatch[0].shape}")
    #     print(f"{vbatch[0].shape}")

    # end = datetime.now()
    # dl.teardown("fit")
    # print(f"Duration: {end - start}")

    dl = UIPRMDSingleDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2)
    for i in range(1, 11):
        for j in range(1, 11):
            dl.set_subject(subject=i)
            dl.set_movement(movement=j)
            dt = iter(dl.train_dataloader())
            dv = iter(dl.val_dataloader())
            for tbatch, vbatch in zip(dt, dv):
                print(f"{tbatch[0].shape}")
            print(f"{vbatch[1].shape}")

