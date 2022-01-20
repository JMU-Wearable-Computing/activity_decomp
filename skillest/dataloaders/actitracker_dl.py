from os.path import join
from typing import Optional, Sequence

from torch._C import FunctionSchema

from skillest.dataloaders import (ACTITRCKER_ACTIVITIES, ACTITRCKER_ACTIVITIES_TO_IDX, 
                                  ACTITRCKER_DIR,
                                  ACTITRCKER_SAMPLE_RATE_PER_SEC)
from skillest.dataloaders import transformations
from skillest.dataloaders.imu_dl import IMUDataModule
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


class ActitrackerDL(IMUDataModule):

    def __init__(self,
                 file_prefix: str = ".", 
                 batch_size: int = 256,
                 num_seconds: int = 10,
                 num_batches: int = 1000,
                 transformations: Optional[Sequence[FunctionSchema]] = None,
                 return_activities: bool = False,
                 return_user: bool = False,
                 **dataloader_kwargs):

        super().__init__(ACTITRCKER_DIR, 
                         ACTITRCKER_ACTIVITIES, 
                         ACTITRCKER_ACTIVITIES_TO_IDX, 
                         ACTITRCKER_SAMPLE_RATE_PER_SEC,
                         file_prefix=file_prefix,
                         batch_size=batch_size,
                         num_seconds=num_seconds,
                         transformations=transformations,
                         return_activities=return_activities,
                         return_user=return_user,
                         num_batches=num_batches,
                         **dataloader_kwargs)

            
if __name__ == "__main__":
    transformations = [
        noise_transform_vectorized,
        scaling_transform_vectorized,
        rotation_transform_vectorized,
        negate_transform_vectorized,
        time_flip_transform_vectorized,
        channel_shuffle_transform_vectorized,
        time_segment_permutation_transform_improved,
        time_warp_transform_improved
    ]
    dl = ActitrackerDL(num_workers=8, transformations=transformations, return_user=True, return_activities=True)
    dl.setup("fit")
    d = iter(dl.train_dataloader())
    from datetime import datetime
    start = datetime.now()
    idx = 0
    while idx < 1000:
        batch = next(d)
        if idx < 2:
            print(f"{idx}: {batch}")
            # print(f"{idx}: {next(batch)}")
        pass
        idx += 1
    print(idx)
    end = datetime.now()
    dl.teardown("fit")
    print(f"Duration: {end - start}")

