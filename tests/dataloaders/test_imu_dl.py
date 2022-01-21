import pytest
import torch
from skillest.dataloaders import ActitrackerDL
import numpy as np
from skillest.dataloaders.transformations import (channel_shuffle_transform_vectorized, 
                                                  get_cubic_spline_interpolation,
                                                  negate_transform_vectorized,
                                                  noise_transform_vectorized,
                                                  rotation_transform_vectorized, 
                                                  scaling_transform_vectorized,
                                                  time_flip_transform_vectorized,
                                                  time_segment_permutation_transform_improved,
                                                  time_warp_transform_improved,
                                                  time_warp_transform_low_cost)
from os.path import join
import pandas as pd
from skillest.utils import get_activity_data_info
from skillest.dataloaders import (ACTITRCKER_ACTIVITIES, 
                                  ACTITRCKER_DIR,
                                  ACTITRCKER_ACTIVITIES_TO_IDX)


@pytest.fixture(scope='module')
def global_data():
    data = {}
    data["train_data"] = pd.read_csv(join(ACTITRCKER_DIR, "train.csv"))
    data["og_dists"] = data["train_data"]["Activity"].value_counts().values
    data["idx_to_activities"] = {v: k for k,v in ACTITRCKER_ACTIVITIES_TO_IDX.items()}

    _, feature_columns = get_activity_data_info(ACTITRCKER_DIR)

    assert not np.any(np.isnan(data["train_data"][feature_columns].values))

    transformations = [
        time_flip_transform_vectorized,
    ]
    data["dl"] = ActitrackerDL(num_workers=0, transformations=transformations, return_user=True, return_activities=True)
    data["dl"].setup("fit")
    return data


def test_dataloader(global_data):
    """Test dataloader to confirm all users and activities are represented.
    """
    d = iter(global_data["dl"].train_dataloader())

    cat_sums = torch.zeros(len(global_data["idx_to_activities"]))
    user_sums = torch.zeros(len(global_data["train_data"]["UserID"].unique()))
    idx = 0
    while idx < 10:
        batch = next(d)
        users, user_counts = torch.unique(batch[1], return_counts=True)
        activities, cat_counts = torch.unique(batch[2], return_counts=True)
        cat_sums += cat_counts
        user_sums += user_counts

        idx += 1

    # Make sure every user and activity is represented
    assert all(cat_sums > 0)
    assert all(user_sums > 0)



def test_nans(global_data):
    """Test dataloader to confirm all users and activities are represented.
    """
    d = iter(global_data["dl"].train_dataloader())

    idx = 0
    while idx < 1000:
        batch = next(d)
        assert not torch.any(torch.isnan(batch[0]))
        assert not torch.any(torch.isnan(batch[1]))
        assert not torch.any(torch.isnan(batch[2]))

        idx += 1


def disabled_test_distributions():
    """This test was created to make sure the distribution of activities was
    the same in the output as in the train data. However this is not the case
    but this is ok. The reason this is different is because the dataset
    iterator uniformly samples authors and then samples from activities
    from each user. So the distribution will be closer to the frequency of
    activities each user has created. 
    """

    train_data = pd.read_csv(join(ACTITRCKER_DIR, "train.csv"))
    og_dists = train_data["Activity"].value_counts().values
    idx_to_activities = {v: k for k,v in ACTITRCKER_ACTIVITIES_TO_IDX.items()}

    transformations = [
        time_flip_transform_vectorized,
    ]
    dl = ActitrackerDL(num_workers=0, transformations=transformations, return_user=True, return_activities=True)
    dl.setup("fit")
    d = iter(dl.train_dataloader())

    cat_sums = torch.zeros(len(idx_to_activities))
    idx = 0
    while idx < 10000:
        batch = next(d)
        activities, counts = torch.unique(batch[2], return_counts=True)
        cat_sums += counts

        idx += 1
    cat_freq = cat_sums / torch.sum(cat_sums)

    og_freqs = og_dists / np.sum(og_dists)
    for cat, (freq, og_freq) in enumerate(zip(cat_freq, og_freqs)):
        print(f"{idx_to_activities[cat]}: {freq}")
        print("Original train dist: {og_freq}")

    dl.teardown("fit")
    assert False