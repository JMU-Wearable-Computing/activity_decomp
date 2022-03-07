import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from torch import nn
import torchmetrics

class UnFlatten(nn.Module):
    def __init__(self, num_filters=256):
        super().__init__()
        self.num_filters =  num_filters

    def forward(self, input):
        return input.view(input.size(0), self.num_filters, 1)


class Deep1DCNNEncoder(pl.LightningModule):

    def __init__(self, input_shape, embedding_size=256, num_filters=256, batch_norm=False, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.C, self.T = input_shape

        encoder = []
        #### CNN1D layer 1 ####
        encoder.append(nn.Conv1d(in_channels=self.C, out_channels=num_filters, 
                                 kernel_size=7, padding="same"))
        encoder.append(nn.ReLU())
        if batch_norm:
            encoder.append(nn.BatchNorm1d(num_filters))
        encoder.append(nn.MaxPool1d(3))
        #### CNN1D layer 1 ####

        #### CNN1D layer 2 ####
        encoder.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=7, padding="same"))
        encoder.append(nn.ReLU())
        if batch_norm:
            encoder.append(nn.BatchNorm1d(num_filters))
        encoder.append(nn.MaxPool1d(3))
        #### CNN1D layer 2 ####

        #### CNN1D layer 3 ####
        encoder.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=3, padding="same"))
        encoder.append(nn.ReLU())
        if batch_norm:
            encoder.append(nn.BatchNorm1d(num_filters))
        #### CNN1D layer 3 ####

        #### CNN1D layer 4 ####
        encoder.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=3, padding="same"))
        encoder.append(nn.ReLU())
        if batch_norm:
            encoder.append(nn.BatchNorm1d(num_filters))
        #### CNN1D layer 4 ####

        #### CNN1D layer 5 ####
        encoder.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=3, padding="same"))
        encoder.append(nn.ReLU())
        if batch_norm:
            encoder.append(nn.BatchNorm1d(num_filters))
        encoder.append(nn.MaxPool1d(3))
        #### CNN1D layer 5 ####

        #### CNN1D layer 6 ####
        encoder.append(nn.Conv1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=1, padding="same"))
        encoder.append(nn.ReLU())
        #### CNN1D layer 6 ####

        encoder.append(nn.Flatten())
        
        self.encoder_base = nn.Sequential(*encoder)
        # Get output shape from CNN, not the ideal way to do this but it works well
        self.encoder_outshape = self.encoder_base(torch.randn([1, self.C, self.T])).shape[-1]

        self.mean_layer = nn.Linear(self.encoder_outshape, embedding_size)
        self.var_layer = nn.Linear(self.encoder_outshape, embedding_size)
    
    def forward(self, x):
        x_hat = self.encoder_base(x)
        mean = self.mean_layer(x_hat)
        var = torch.exp(self.var_layer(x_hat) / 2)
        return mean, var


class Deep1DCNNDecoder(pl.LightningModule):

    def __init__(self, output_shape, base_encoder_outshape, embedding_size=256, num_filters=256, batch_norm=False, lr=0.001):
        C, T = output_shape
        super().__init__()

        decoder = []

        decoder.append(nn.Linear(embedding_size, base_encoder_outshape))
        decoder.append(UnFlatten(base_encoder_outshape))

        #### CNN1D layer 6 ####
        decoder.append(nn.ConvTranspose1d(in_channels=base_encoder_outshape, out_channels=num_filters, 
                                 kernel_size=1))
        decoder.append(nn.ReLU())
        #### CNN1D layer 6 ####

        #### CNN1D layer 5 ####
        decoder.append(nn.ConvTranspose1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=3))
        decoder.append(nn.ReLU())
        if batch_norm:
            decoder.append(nn.BatchNorm1d(num_filters))
        decoder.append(nn.MaxUnpool1d(3))
        #### CNN1D layer 5 ####

        #### CNN1D layer 4 ####
        decoder.append(nn.ConvTranspose1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=3))
        decoder.append(nn.ReLU())
        if batch_norm:
            decoder.append(nn.BatchNorm1d(num_filters))
        #### CNN1D layer 4 ####

        #### CNN1D layer 3 ####
        decoder.append(nn.ConvTranspose1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=3))
        decoder.append(nn.ReLU())
        if batch_norm:
            decoder.append(nn.BatchNorm1d(num_filters))
        #### CNN1D layer 3 ####

        #### CNN1D layer 2 ####
        decoder.append(nn.ConvTranspose1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=7))
        decoder.append(nn.ReLU())
        if batch_norm:
            decoder.append(nn.BatchNorm1d(num_filters))
        decoder.append(nn.MaxUnpool1d(3))
        #### CNN1D layer 2 ####

        #### CNN1D layer 1 ####
        decoder.append(nn.ConvTranspose1d(in_channels=num_filters, out_channels=num_filters, 
                                 kernel_size=7))
        decoder.append(nn.ReLU())
        if batch_norm:
            decoder.append(nn.BatchNorm1d(num_filters))
        decoder.append(nn.MaxUnpool1d(3))
        #### CNN1D layer 1 ####

        #### CNN1D layer output ####
        decoder.append(nn.ConvTranspose1d(in_channels=num_filters, out_channels=C,
                                 kernel_size=1))
        #### CNN1D layer output ####

        self.decoder_base = nn.Sequential(*decoder)
    

    def forward(self, x):
        x_hat = self.decoder_base(x)
        return x_hat.reshape([-1, self.C, self.T])

