import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from skillest.dataloaders import UIPRMDSingleDataloader
from skillest.models import SVM
from skillest.utils import SubjectMovementLogger


dl = UIPRMDSingleDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2, 
                            points_to_use=["right_upper_leg",
                                           "left_upper_leg",
                                           ])


tb_logger = pl_loggers.TensorBoardLogger("logs/")
wandb_logger = pl_loggers.WandbLogger(  # name="svm_test",
    project="UI-PRMD",
    entity="jmu-wearable-computing",
    save_dir="logs/",
    log_model=True)
wandb_logger.experiment.config["type"] = "SVM"
for hparam, v in dl.hparams.items():
    print(f"{hparam}: {v}")
    wandb_logger.experiment.config[hparam] = v if v is not None else "None"

submov_logger = SubjectMovementLogger(num_mov=1, cols=["All movements"])
ks = SVM(custom_logger=submov_logger, C=1.0, kernel="linear")

for i in range(1, 11):
    print(f"Subject: {i}")
    wandb_logger._prefix = f"sub{i}"
    trainer = pl.Trainer(max_steps=1,
                        val_check_interval=1.0,
                        limit_train_batches=1,
                        limit_val_batches=1.0,
                        num_sanity_val_steps=0,
                        # logger=[tb_logger, wandb_logger],
                        log_every_n_steps=1)
    dl.set_subject(subject=i)
    submov_logger.set_subject_movement(subject=i, movement=1)

    train_loader = iter(dl.train_dataloader())
    val_loader = iter(dl.val_dataloader())

    trainer.fit(ks, train_loader, val_loader)
submov_logger.reduce()
