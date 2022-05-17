import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
import numpy as np
import torch

from skillest.dataloaders import UIPRMDSingleDataloader
from skillest.models import SVM
from skillest.utils import SubjectMovementLogger

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

transforms = transforms.Compose([np.array,
                                 noise_transform_vectorized,
                                 scaling_transform_vectorized,
                                 time_flip_transform_vectorized,
                                 time_segment_permutation_transform_improved,
                                 time_warp_transform_low_cost,
                                 torch.from_numpy])

dl = UIPRMDSingleDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2, 
                            points_to_use=["head", "neck", "chest", "spine", "right_upper_leg",
                                           "right_lower_leg", "right_foot", "left_upper_leg",
                                           "left_lower_leg", "left_foot"],
                            transforms=transforms,
                            data_repetitions=5)


tb_logger = pl_loggers.TensorBoardLogger("logs/")
wandb_logger = pl_loggers.WandbLogger(  # name="svm_test",
    project="UI-PRMD", entity="jmu-wearable-computing",
    save_dir="logs/",
    log_model=True)
wandb_logger.experiment.config["type"] = "SVM"
for hparam, v in dl.hparams.items():
    print(f"{hparam}: {v}")
    wandb_logger.experiment.config[hparam] = v if v is not None else "None"

submov_logger = SubjectMovementLogger()
ks = SVM(custom_logger=submov_logger, C=1.0, kernel="linear")

for i in range(1, 11):
    for j in range(1, 11):
        print(f"Subject: {i}, Movement: {j} ")
        wandb_logger._prefix = f"sub{i}_mov{j}"
        trainer = pl.Trainer(max_steps=1,
                            val_check_interval=1.0,
                            limit_train_batches=1,
                            limit_val_batches=1.0,
                            num_sanity_val_steps=0,
                            # logger=[tb_logger, wandb_logger],
                            log_every_n_steps=1)
        dl.set_subject(subject=i)
        dl.set_movement(movement=j)
        submov_logger.set_subject_movement(subject=i, movement=j)

        train_loader = iter(dl.train_dataloader())
        val_loader = iter(dl.val_dataloader())

        trainer.fit(ks, train_loader, val_loader)
submov_logger.reduce()