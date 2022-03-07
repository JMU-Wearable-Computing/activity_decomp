import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from torch import nn
import torch.nn.functional as F
import torchmetrics
from skillest.models import Deep1DCNNDecoder, Deep1DCNNEncoder, Shallow1DCNNEncoder, Shallow1DCNNDecoder


class VAE(pl.LightningModule):

    def __init__(self, encoder, decoder, z_dim, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.type = "VAE"
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        h_dim = encoder(torch.randn([1, C, T])).shape[-1]
        self.hparams["h_dim"] = h_dim

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
    
    def reparametarize(self, mu, logvar):
        return mu + torch.randn([*logvar.shape]) * torch.exp(logvar / 2)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparametarize(mu, logvar)
        h_hat = self.fc3(z).reshape(h.shape)
        return self.decoder(h_hat), mu, logvar

    def training_step(self, x, batch_idx):
        assert decoder.validate_length(x.shape[-1])
        x_hat, mu, logvar = self(x)
        loss = self.loss_fn(x, x_hat, mu, logvar)

        # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}
    
    def loss_fn(self, x, recon_x, mu, logvar):
        BCE = F.mse_loss(recon_x, x, reduction="sum")

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
        return BCE + KLD
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    x = torch.randn([10, 3, 127])
    B, C, T = x.shape
    encoder = Shallow1DCNNEncoder(C)
    decoder = Shallow1DCNNDecoder(C)
    vae = VAE(encoder, decoder, z_dim=256)
    x_hat, mean, logvar = vae.forward(x)
    print(x_hat.shape)
    vae.training_step(x, None)
