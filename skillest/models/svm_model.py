import torch
from sklearn.svm import SVC
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
import wandb
from skillest.dataloaders import ActitrackerDL
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


class SVM(pl.LightningModule):

    def __init__(self, gamma="auto", C=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.svm = SVC(gamma=gamma, C=C, probability=True)
        self.type = "svm"

    def forward(self, x):
        return self.svm.predict(x)

    def backward(self, *args):
        return torch.tensor(0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        self.svm.fit(x, y)
        accuracy = torch.tensor(self.svm.score(x, y))
        return {"loss": torch.tensor(0), "train_accuracy": accuracy}
    
    def training_step_end(self, outs):
        self.log("train_accuracy", outs["train_accuracy"])

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        accuracy = torch.tensor(self.svm.score(x, y))
        return {"loss": torch.tensor(0), "accuracy": accuracy}
    
    def validation_step_end(self, outs):
        self.log("val_accuracy", outs["accuracy"])

    def configure_optimizers(self):
        return None


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
    dl = ActitrackerDL(num_workers=8, transformations=transformations, return_activities=True)
    dl.setup("fit")
    train_loader = dl.train_dataloader()
    val_loader = dl.val_dataloader()

    svm = SVM(C=2.0, gamma="scale")
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    wandb_logger = pl_loggers.WandbLogger(#name="svm_test", 
                                          project="test", 
                                          entity="jmu-wearable-computing",
                                          save_dir="logs/",
                                          log_model=True)
    # wandb_logger.watch(svm)
    # run = wandb_logger.experiment
    # model_artifact = run.Artifact("svm", type="svm")
    # model_artifact.add_dir("logs/")
    # run.log_artifact(model_artifact)

    trainer = pl.Trainer(max_steps=2,
                         val_check_interval=1.0,
                         limit_train_batches=1, 
                         limit_val_batches=1, 
                         num_sanity_val_steps=0,
                         logger = [tb_logger, wandb_logger],
                         log_every_n_steps=1)
    trainer.fit(svm, train_loader, val_loader)

