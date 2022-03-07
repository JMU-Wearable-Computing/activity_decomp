import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from torch import nn
import torchmetrics


class Shallow1DCNNEncoder(pl.LightningModule):
    # Input must be length s.t. (x-31)%16 == 0 with kernel=3, stride=2.
    def __init__(self, in_channels, num_filters=256):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, num_filters // 2, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(num_filters // 2, num_filters, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv1d(num_filters, num_filters, kernel_size=3, stride=2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)


class Shallow1DCNNDecoder(pl.LightningModule):
    def __init__(self, in_channels, num_filters=256):
        super().__init__()
        self.save_hyperparameters()

        self.decoder = nn.Sequential(
            # UnFlatten(h_dim),
            nn.ConvTranspose1d(num_filters, num_filters, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, num_filters, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, num_filters // 2, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters // 2, in_channels, kernel_size=3, stride=2),
        )

    def forward(self, x):
        return self.decoder(x) 
    
    def validate_length(self, length: int) -> bool:
        """Ensures length 

        Args:
            length (int): The length of the time dimension of the data.

        Returns:
            bool: If this length is valid for the decoder to return the same length.
        """
        # Input must be length s.t. (x-31)%16 == 0 with kernel=3, stride=2.
        return (length - 31) % 16 == 0

