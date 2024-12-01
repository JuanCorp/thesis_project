import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl


class DVAE(pl.LightningModule):
    def __init__(self,
                 embedding_size,
                 topic_size,
                 beta=2.0):
        super().__init__()

        self.embedding_size = embedding_size
        self.topic_size = topic_size
        self.beta = beta

        self.encoder = nn.Sequential(
           nn.Linear(embedding_size, 16),
           nn.ReLU(),
           nn.Linear(16, topic_size),
           nn.ReLU()
       )
        self.decoder = nn.Sequential(
           nn.Linear(topic_size, 16),
           nn.ReLU(),
           nn.Linear(16, embedding_size)
       )


    def forward(self, x):
        alpha = self.encoder(x)
        x_recon = self.decoder(alpha)
        return x_recon

    def training_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        recon = self.objective(x, x_recon)
        loss = recon
        self.log_dict({'train/loss': loss,
                       'train/recon': recon},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        recon = self.objective(x, x_recon)
        loss = recon 
        self.log_dict({'val/loss': loss,
                       'val/recon': recon},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def objective(self, x, x_recon):
        recon = -torch.sum(x * x_recon, dim=1).mean()
        return recon
    

    def predict(self,x):
        alpha = F.softplus(self.encoder_norm(self.encoder(x)))
        alpha = torch.max(torch.tensor(0.00001, device=x.device), alpha)
        dist = Dirichlet(alpha)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        keep = torch.argmax(z,dim=1)
        assigned_topics = [[a] for a in keep]
        return assigned_topics
