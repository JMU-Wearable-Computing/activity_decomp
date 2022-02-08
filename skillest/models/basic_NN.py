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
import torch
from torch import nn
import torchmetrics


class BasicNN(pl.LightningModule):

    def __init__(self, input_shape, output_shape, lr=0.001, layer_size=30, loss_fn=nn.CrossEntropyLoss()):
        super().__init__()
        self.save_hyperparameters()
        self.type = "BasicNN"
        self.lr = lr
        self.loss_fn = loss_fn
        self.train_accuracy = torchmetrics.Accuracy() 
        self.val_accuracy = torchmetrics.Accuracy() 

        self.l1 = nn.Linear(input_shape, 30)
        self.l2 = nn.Linear(30, 30)
        self.l3 = nn.Linear(30, output_shape)

    def forward(self, x):
        x_hat = torch.relu(self.l1(x))
        x_hat = torch.relu(self.l2(x_hat))
        x_hat = self.l3(x_hat)
        return torch.softmax(x_hat, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.train_accuracy(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "train_accuracy": self.train_accuracy}
    
    # def training_step_end(self, outs):
    #     self.log("train_step_accuracy", outs["train_accuracy"])

    # def training_epoch_end(self, outs):
    #     self.log("train_epoch_accuracy", self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.shape[0], -1)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.val_accuracy(y_hat, y)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "val_accuracy": self.val_accuracy}
    
    # def validation_step_end(self, outs):
    #     self.log("val_step_accuracy", outs["val_accuracy"])

    # def validation_epoch_end(self, outs):
    #     self.log("val_epoch_accuracy", self.val_accuracy.compute())

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

if __name__ == '__main__':
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
    dl = ActitrackerDL(num_workers=4, transformations=transformations, return_activities=True)
    dl.setup("fit")

    # Batch, time step, channels
    B, T, C = dl.data_shape
    model = BasicNN(T * C, dl.num_activities, layer_size=300)
    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    wandb_logger = pl_loggers.WandbLogger(#name="svm_test", 
                                            project="test", 
                                            entity="jmu-wearable-computing",
                                            save_dir="logs/",
                                            log_model=True)
    wandb_logger.experiment.config["transformations"] = transformations

    trainer = pl.Trainer(max_epochs=2,
                            val_check_interval=1.0,
                            limit_train_batches=1000, 
                            limit_val_batches=500, 
                            num_sanity_val_steps=1,
                            logger = [tb_logger, wandb_logger],
                            log_every_n_steps=1)
    trainer.fit(model, datamodule=dl)