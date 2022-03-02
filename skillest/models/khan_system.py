from matplotlib import use
import numpy as np
import torch
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from torchvision.datasets import MNIST
import scipy.stats


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import os
import wandb

from skillest.dataloaders import ActitrackerDL, UIPRMDDataloader
from tslearn.piecewise import SymbolicAggregateApproximation

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


class KhanSystem(pl.LightningModule):

    def __init__(self,
                 descretizer_kwargs: dict,
                 high_level_model=SVC,
                 high_level_model_kwargs={
                     "gamma": "auto", "C": 1.0, "cache_size": 2000},
                 low_level_classifier=SVC,
                 low_level_classifier_kwargs={
                     "gamma": "auto", "C": 1.0, "cache_size": 2000},
                 descretizer=SymbolicAggregateApproximation, 
                 use_decision_func: bool = False):
        super().__init__()
        self.save_hyperparameters()

        self.high_level_model = high_level_model(**high_level_model_kwargs)
        self.low_level_classifier = low_level_classifier(
            **low_level_classifier_kwargs)

        self.use_decision_func = use_decision_func

        assert "n_segments" in descretizer_kwargs, "missing n_paa_segments in descretizer_kwargs."

        self.descretizer = descretizer(**descretizer_kwargs)

    def forward(self, x):
        return self.svm.predict(x)

    def backward(self, *args):
        return torch.tensor(0)

    def segment_data(self, x, n_segments):
        """
            Returns data in the shape of (B, n_segments, T / n_segments * C).
            Pads if needed.
            This is the same behavior of the sax implementation and should result
            in the same segments: https://github.com/tslearn-team/tslearn/blob/42a56cce63d8263982d616fc2bef9009ccbedab4/tslearn/piecewise/piecewise.py#L146
        Args:
            x ([type]): [description]
            n_segments ([type]): [description]

        Returns:
            [type]: [description]
        """
        b, t, c = x.shape
        # pad x so that we can evenly divide t into n_segments
        pad = n_segments - t % n_segments
        x_padded = np.pad(x, [(0, 0), (0, pad), (0, 0)])
        assert int((t + pad) / n_segments) == (t + pad) / n_segments
        return x_padded.reshape([b, n_segments, int((t + pad) / n_segments), c])

    def gen_low_level_features(self, x: np.array):
        """Generates the low level features

        Args:
            x (np.array): Must be of shape (B, n_segments, (T + pad) / n_segments, C)
        """
        B, n_seg, t, C = x.shape
        # We must transpose to apply our operations on the time dimension.
        # Each statistic can only be computed for a single channel on
        # the time dimension.
        x_t = x.transpose([0, 1, 3, 2])

        # ECDF Inverse
        # magic number 20 from khan paper
        num_ecdf_features = 20
        quantiles = np.linspace(0, 1, num=num_ecdf_features) * (t - 1)
        sorted = np.sort(x_t, axis=3)
        ecdf_features = sorted[..., quantiles.astype(np.int32)]

        # To find entropy we must construct a histogram over our
        # timesteps. We use 10 bins
        entropy = np.zeros([B, n_seg, C, 1])
        for b in range(B):
            for s in range(n_seg):
                for c in range(C):
                    hist, _ = np.histogram(x_t[b, s, c, :], bins=10)
                    _, counts = np.unique(hist, return_counts=True)
                    entropy[b, s, c] = scipy.stats.entropy(counts)

        energy = np.expand_dims(
            np.einsum("ijkl,ijkl->ijk", x_t, x_t) / t, axis=3)

        mean = np.mean(x_t, axis=3, keepdims=True)
        std = np.std(x_t, axis=3, keepdims=True)
        fft = np.fft.fft(x_t, axis=3).view(np.float32)
        # cor_coef ??? no idea what khan means this to be

        # staistic: [B, n_segments, C, ?]
        return np.concatenate([ecdf_features, energy, entropy, mean, std], axis=3)

    def training_step(self, batch, batch_idx):
        x, skill_labels = batch
        B, T, C = x.shape

        # x: (B, T, C)
        low_level_labels = self.descretizer.fit_transform(x)
        _, n_segments, _ = low_level_labels.shape
        low_level_labels = low_level_labels.reshape(B * n_segments * C)
        # low_level_labels: (B * n_segments * C)
        x_seg = self.segment_data(x, n_segments)
        # (B, n_segments, (T + pad) / n_segments, C)
        x_features = self.gen_low_level_features(x_seg)
        # (B, n_segments, C, # of features)
        x_features = x_features.reshape([B * n_segments * C, -1])
        # (B * n_segments * C, # of features)
        # now each row is a low level sample to be predicted by our svm
        # a low level sample is defined as low level features associated
        # with a channel, segments, and high level sample

        assert np.count_nonzero(
            np.isnan(x_features)) == 0, "Nans in x_features"

        print("Starting low level classifier")
        self.low_level_classifier.fit(x_features, low_level_labels)
        if not self.use_decision_func:
            confidence_scores = self.low_level_classifier.predict_proba(
                x_features)
        else:
            confidence_scores = self.low_level_classifier.decision_function(
                x_features)
        # x: (B * n_segments * C, alphabet_size)
        confidence_scores = confidence_scores.reshape([B, n_segments, -1])
        # x: (B, n_segments, C * alphabet_size)
        confidence_scores_t = confidence_scores.transpose([0, 2, 1])

        # Find the statisical features per sample
        mean = np.mean(confidence_scores_t, axis=2)
        std = np.std(confidence_scores_t, axis=2)
        var = np.var(confidence_scores_t, axis=2)
        median = np.median(confidence_scores_t, axis=2)
        # statistic: [B, C * alphabet_size]
        high_level_features = np.concatenate([mean, std, var, median], axis=1)
        # features: [B, C * alphabet_size * 4]

        print("Starting high level classifier")
        self.high_level_model.fit(high_level_features, skill_labels)

        accuracy = torch.tensor(self.high_level_model.score(
            high_level_features, skill_labels))
        print(f"train acc: {accuracy}")
        return {"loss": torch.tensor(0), "train_accuracy": accuracy}

    def training_step_end(self, outs):
        self.log("train_accuracy", outs["train_accuracy"])

    def validation_step(self, batch, batch_idx):
        x, skill_labels = batch
        B, T, C = x.shape

        # x: (B, T, C)
        low_level_labels = self.descretizer.transform(x)
        _, n_segments, _ = low_level_labels.shape
        low_level_labels = low_level_labels.reshape(B * n_segments * C)
        # low_level_labels: (B * n_segments * C)
        x_seg = self.segment_data(x, n_segments)
        # (B, n_segments, (T + pad) / n_segments, C)
        x_features = self.gen_low_level_features(x_seg)
        # (B, n_segments, C, # of features)
        x_features = x_features.reshape([B * n_segments * C, -1])
        # (B * n_segments * C, # of features)

        assert np.count_nonzero(
            np.isnan(x_features)) == 0, "Nans in x_features"

        print("Starting low level classifier")
        if not self.use_decision_func:
            confidence_scores = self.low_level_classifier.predict_proba(
                x_features)
        else:
            confidence_scores = self.low_level_classifier.decision_function(
                x_features)
        # x: (B * n_segments * C, alphabet_size)
        confidence_scores = confidence_scores.reshape([B, n_segments, -1])
        # x: (B, n_segments, C * alphabet_size)
        confidence_scores_t = confidence_scores.transpose([0, 2, 1])

        # Find the statisical features per sample
        mean = np.mean(confidence_scores_t, axis=2)
        std = np.std(confidence_scores_t, axis=2)
        var = np.var(confidence_scores_t, axis=2)
        median = np.median(confidence_scores_t, axis=2)
        # statistic: [B, C * alphabet_size]
        high_level_features = np.concatenate([mean, std, var, median], axis=1)
        # features: [B, C * alphabet_size * 4]

        print("Starting high level classifier")

        accuracy = torch.tensor(self.high_level_model.score(
            high_level_features, skill_labels))
        print(f"val acc: {accuracy}")

        return {"loss": torch.tensor(0), "accuracy": accuracy}

    def validation_step_end(self, outs):
        self.log("val_accuracy", outs["accuracy"])

    def configure_optimizers(self):
        return None


if __name__ == "__main__":
    # transformations = [
    # ]
    # dl = ActitrackerDL(batch_size=10000, num_batches=1,
    #                    num_workers=8, return_activities=True, transformations=transformations)
    dl = UIPRMDDataloader(batch_size=-1, num_ep_in_train=6, num_ep_in_val=2, num_ep_in_test=2)
    dl.setup("fit")
    train_loader = dl.train_dataloader()
    val_loader = dl.val_dataloader()

    sax_params = {"n_segments": 20, "alphabet_size_avg": 5, "scale": True}
    high_level_model_kwargs = {}
    low_level_classifier_kwargs = {}

    ks = KhanSystem(sax_params, 
                    high_level_model=RandomForestClassifier,
                    low_level_classifier=RandomForestClassifier,
                    high_level_model_kwargs=high_level_model_kwargs,
                    low_level_classifier_kwargs=low_level_classifier_kwargs)

    tb_logger = pl_loggers.TensorBoardLogger("logs/")
    wandb_logger = pl_loggers.WandbLogger(  # name="svm_test",
        project="test",
        entity="jmu-wearable-computing",
        save_dir="logs/",
        log_model=True)
    wandb_logger.experiment.config["type"] = "khan_system"
    for hparam, v in dl.hparams.items():
        print(f"{hparam}: {v}")
        wandb_logger.experiment.config[hparam] = v if v is not None else "None"

    # wandb_logger.watch(ks)
    # run = wandb_logger.experiment
    # model_artifact = run.Artifact("svm", type="svm")
    # model_artifact.add_dir("logs/")
    # run.log_artifact(model_artifact)

    trainer = pl.Trainer(max_steps=1,
                         val_check_interval=1.0,
                         limit_train_batches=1,
                         limit_val_batches=1.0,
                         num_sanity_val_steps=0,
                         logger=[tb_logger, wandb_logger],
                         log_every_n_steps=1)
    trainer.fit(ks, train_loader, val_loader)

    # sax_params = {"n_segments": 20, "alphabet_size_avg": 5, "scale": True}
    # ks = KhanSystem(sax_params)

    # x = np.random.rand(10, 200, 3)
    # skill_labels = np.random.randint(2, size=[10])
    # data = [x, skill_labels]
    # print(ks.training_step(data, 0))
