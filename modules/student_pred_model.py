import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl


class Student_Pred(pl.LightningModule):
    def __init__(self,
                 input_size,
                 topic_size,):
        super().__init__()
        self.input_size = input_size
        self.topic_size = topic_size

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=100),
            nn.Softplus(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=100, out_features=100),
            nn.Softplus(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=100,out_features=self.topic_size)

        )

        

        # save hyperparameters
        self.save_hyperparameters()


    def forward(self, x):
        pred_y = F.softmax(self.encoder(x),dim=1)
        return pred_y

    def training_step(self, batch, batch_idx):
        x,y = batch
        pred = self(x)
        log_loss = self.objective(pred,y)
        loss = log_loss
        self.log_dict({'train/loss': loss},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        pred = self(x)
        log_loss = self.objective(pred,y)
        loss = log_loss
        self.log_dict({'val/loss': loss},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    

    def objective(self, pred,y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred,y)
        return loss
    

    def predict(self,x):
        x_recon, mu, logvar, gauss_z, dir_z = self(x)
        z = dir_z
        keep = torch.argmax(z,dim=1)
        assigned_topics = [[a] for a in keep]
        return assigned_topics
