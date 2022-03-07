ACTITRCKER_DIR = "activities/Actitracker"
ACTITRCKER_XML_FILEPATH = "activities/Actitracker/Actitracker.xml"
ACTITRCKER_SAMPLE_RATE_PER_SEC = 20
ACTITRCKER_ACTIVITIES = ["Walking", "Jogging", "Upstairs", "Downstairs", "Sitting", "Standing"]
ACTITRCKER_ACTIVITIES_TO_IDX = {a: i for i, a in enumerate(ACTITRCKER_ACTIVITIES)}


from skillest.dataloaders.actitracker_dl import ActitrackerDL
from skillest.dataloaders.ui_prmd_dl import UIPRMDDataloader
from skillest.dataloaders.imu_dl import IMUDataModule, IMUDataset
from skillest.dataloaders.hdf5_dl import H5DataModule

from skillest.dataloaders.transformations import (channel_shuffle_transform_vectorized, force_to_shape, 
                                                  get_cubic_spline_interpolation,
                                                  negate_transform_vectorized,
                                                  noise_transform_vectorized,
                                                  rotation_transform_vectorized, 
                                                  scaling_transform_vectorized,
                                                  time_flip_transform_vectorized,
                                                  time_segment_permutation_transform_improved,
                                                  time_warp_transform_improved,
                                                  time_warp_transform_low_cost,
                                                  sample_windows,
                                                  pad_to_shape,
                                                  force_to_shape)