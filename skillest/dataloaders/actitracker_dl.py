from os.path import join

from skillest.dataloaders import (ACTITRCKER_ACTIVITIES, 
                                  ACTITRCKER_DIR,
                                  ACTITRCKER_SAMPLE_RATE_PER_SEC)
from skillest.dataloaders.imu_dl import IMUDataModule


class ActitrackerDL(IMUDataModule):

    def __init__(self,
                 file_prefix: str = ".", 
                 batch_size: int = 256,
                 num_seconds: int = 10,
                 num_batches: int = 1000,
                 return_activities: bool = False,
                 return_user: bool = False,
                 **dataloader_kwargs):

        super().__init__(ACTITRCKER_DIR, 
                         ACTITRCKER_ACTIVITIES, 
                         ACTITRCKER_SAMPLE_RATE_PER_SEC,
                         file_prefix=file_prefix,
                         batch_size=batch_size,
                         num_seconds=num_seconds,
                         return_activities=return_activities,
                         return_user=return_user,
                         num_batches=num_batches)

            
if __name__ == "__main__":
    dl = ActitrackerDL(num_workers=8, return_user=True, return_activities=True)
    dl.setup("fit")
    d = dl.train_dataloader()
    from datetime import datetime
    start = datetime.now()
    for idx, sample in enumerate(d):
        if idx == 0:
            print(f"{idx}: {sample}")
        pass
    end = datetime.now()
    dl.teardown("fit")
    print(f"Duration: {end - start}")

