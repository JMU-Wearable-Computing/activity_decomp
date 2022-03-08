import pytest
import torch
from skillest.dataloaders import H5DataModule
from skillest.dataloaders.hdf5_dl import H5Dataset
from skillest.dataloaders import force_to_shape, sample_windows
import h5py
import os

TEST_FILE1_NAME = "test1.h5"
TEST_FILE2_NAME = "test2.h5"
GROUP_NAME = "g1"
DATASET_NAME1 = "d1"
DATASET_NAME2 = "d2"
DATASET_NAME3 = "d3"

transforms = [lambda x: x**2]
shapes = [[127, 127], [111, 127], [99, 127]]
batch_size = 64


@pytest.fixture(scope='session', autouse=True)
def hdf5_setup():
    f = h5py.File(TEST_FILE1_NAME, "w")
    f.create_dataset(DATASET_NAME1, [127, 127])
    f.create_dataset(DATASET_NAME2, [111, 127])
    f.create_dataset(DATASET_NAME3, [99, 127])
    f.close()

    f = h5py.File(TEST_FILE2_NAME, "w")
    g = f.create_group(GROUP_NAME)
    g.create_dataset(DATASET_NAME1, [127, 127])
    g.create_dataset(DATASET_NAME2, [111, 127])
    g.create_dataset(DATASET_NAME3, [99, 127])
    f.close()

    yield
    os.remove(TEST_FILE1_NAME)
    os.remove(TEST_FILE2_NAME)



def test_file1_dataset():
    d = H5Dataset(TEST_FILE1_NAME, transforms=transforms)
    d = iter(d)
    for i in range(5):
        next(d)

def test_file2_dataset():
    d = H5Dataset(TEST_FILE2_NAME, transforms=transforms)
    d = iter(d)
    for i in range(5):
        next(d)

def local_force_to_shape(x):
    return force_to_shape(x, [111, -1])

def local_sample_windows(x):
    return sample_windows(x, 20)
    
def test_file1_datamodule():
    dl = H5DataModule(TEST_FILE1_NAME, 
                      transforms=[local_force_to_shape, local_sample_windows],
                      num_workers=2, 
                      num_iterations=2*batch_size, 
                      batch_size=batch_size)

    for batch in dl.train_dataloader():
        assert list(batch.shape) == [batch_size, 20, 127]

def test_file2_datamodule():
    dl = H5DataModule(TEST_FILE2_NAME, 
                      transforms=[local_force_to_shape, local_sample_windows],
                      num_workers=2, 
                      num_iterations=2*batch_size, 
                      batch_size=batch_size)
    dt = dl.train_dataloader()

    for batch in dt:
        assert list(batch.shape) == [batch_size, 20, 127]

