from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import itertools

from skillest.dataloaders import UIPRMDSingleDataloader
from skillest.models import KhanSystem
from skillest.utils import SubjectMovementLogger


dl = UIPRMDSingleDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2, 
                            points_to_use=["head", "neck", "chest", "spine", "right_upper_leg",
                                           "right_lower_leg", "right_foot", "left_upper_leg",
                                           "left_lower_leg", "left_foot"])

sax_params = {"n_segments": 20, "alphabet_size_avg": 5, "scale": True}
high_level_model_kwargs = {"alpha": 20}
low_level_classifier_kwargs = {}

tb_logger = pl_loggers.TensorBoardLogger("logs/")
wandb_logger = pl_loggers.WandbLogger(  # name="svm_test",
    project="UI-PRMD",
    entity="jmu-wearable-computing",
    save_dir="logs/",
    log_model=True)
wandb_logger.experiment.config["type"] = "khan_system"
for hparam, v in dl.hparams.items():
    print(f"{hparam}: {v}")
    wandb_logger.experiment.config[hparam] = v if v is not None else "None"

submov_logger = SubjectMovementLogger()
ks = KhanSystem(sax_params, 
                high_level_model=RidgeClassifier,
                low_level_classifier=RandomForestClassifier,
                high_level_model_kwargs=high_level_model_kwargs,
                low_level_classifier_kwargs=low_level_classifier_kwargs,
                custom_logger=submov_logger)

for i in range(1, 11):
    for j in range(1, 11):
        print(f"Subject: {i}, Movement: {j} ")
        # wandb_logger._prefix = f"sub{i}_mov{j}"
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
