import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def calc_epsilon(p, alpha):
    sqrt_alpha = torch.sqrt(9 * alpha - 3)
    powza = torch.pow(p / (alpha - 1 / 3), 1 / 3)
    return sqrt_alpha * (powza - 1)


def gamma_h_boosted(epsilon, u, alpha):
    """
    Reparameterization for gamma rejection sampler with shape augmentation.
    """
    B = u.shape[0]
    K = alpha.shape[1]
    r = torch.arange(B, device=alpha.device)
    rm = torch.reshape(r, (-1, 1, 1)).float()
    alpha_vec = torch.tile(alpha, (B, 1)).reshape((B, -1, K)) + rm
    u_pow = torch.pow(u, 1. / alpha_vec) + 1e-10
    return torch.prod(u_pow, axis=0) * gamma_h(epsilon, alpha + B)


def gamma_h(eps, alpha):
    b = alpha - 1 / 3
    c = 1 / torch.sqrt(9 * b)
    v = 1 + (eps * c)
    return b * (v ** 3)


def rsvi(alpha):
    B = 10
    gam = torch.distributions.Gamma(alpha + B, 1).sample().to(alpha.device)
    eps = calc_epsilon(gam, alpha + B).detach().to(alpha.device)
    u = torch.rand((B, alpha.shape[0], alpha.shape[1]), device=alpha.device)
    doc_vec = gamma_h_boosted(eps, u, alpha)
    # normalize
    gam = doc_vec
    doc_vec = gam / torch.reshape(torch.sum(gam, dim=1), (-1, 1))
    z = doc_vec.reshape(alpha.shape)
    return z


class DVAE(pl.LightningModule):
    def __init__(self,
                 input_size,
                 embedding_size,
                 topic_size,
                 beta=1.0):
        super().__init__()

        self.embedding_size = embedding_size
        self.topic_size = topic_size
        self.input_size = input_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.embedding_size, out_features=100),
            nn.Softplus(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=100, out_features=100),
            nn.Softplus(),
            nn.Dropout(p=0.2),

            nn.Linear(in_features=100, out_features=self.topic_size),
        )
        #CTM data set+
        #LayerNorm
        self.encoder_norm = nn.BatchNorm1d(num_features=self.topic_size, affine=False)

        # decoder
        self.decoder = nn.Linear(in_features=self.topic_size, out_features=self.input_size)
        self.decoder_norm = nn.BatchNorm1d(num_features=self.input_size, affine=False)

        # save hyperparameters
        self.save_hyperparameters()

    def forward(self, x):
        alpha = F.softplus(self.encoder_norm(self.encoder(x)))
        alpha = torch.max(torch.tensor(0.00001, device=alpha.device), alpha)
        z = rsvi(alpha)
        x_recon = F.softmax(self.decoder_norm(self.decoder(z)), dim=1)
        return x_recon, alpha

    def training_step(self, batch, batch_idx):
        x = batch['X_contextual']
        X_bow = batch["X_bow"]
        X_bow = X_bow.reshape(X_bow.shape[0], -1)
        x_recon, alpha = self(x)
        recon, kl = self.objective(X_bow, x_recon, alpha)
        loss = recon + kl
        loss = loss.sum()
        self.log_dict({'train/loss': loss,
                       'train/recon': recon.sum(),
                       'train/kl': kl.sum()},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['X_contextual']
        X_bow = batch["X_bow"]
        X_bow = X_bow.reshape(X_bow.shape[0], -1)
        x_recon, alpha = self(x)
        recon, kl = self.objective(X_bow, x_recon, alpha)
        loss = recon + kl
        loss = loss.sum()
        self.log_dict({'val/loss': loss,
                       'val/recon': recon.sum(),
                       'val/kl': kl.sum()},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    

    def kl_divergence(self, alpha):
        alpha_prior = torch.ones(alpha.shape, device=alpha.device) * 0.02
        prior = Dirichlet(alpha_prior)
        posterior = Dirichlet(alpha)
        return self.beta * torch.distributions.kl.kl_divergence(posterior, prior)

    def objective(self, x, x_recon, alpha):
        recon = -torch.sum(x * torch.log(x_recon + 1e-10), dim=1)
        print(recon.shape)
        print(recon)
        #criterion = nn.MSELoss(reduction='mean')
        #recon = criterion(x,x_recon)
        kl = self.kl_divergence(alpha)
        return recon, kl

    def get_doc_topic_distribution(self, dataset, n_samples=20):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of sample to collect to estimate the final distribution (the more the better).
        """
        self.eval()

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False
        )
        final_thetas = []
        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X_contextual = batch_samples["X_contextual"]
                X_contextual = X_contextual
                self.zero_grad()
                alpha = F.softplus(self.encoder_norm(self.encoder(X_contextual)))
                alpha = torch.max(torch.tensor(0.00001, device=alpha.device), alpha)
                thetas = rsvi(alpha).cpu().numpy()
                final_thetas.append(thetas)
        return np.concatenate(final_thetas, axis=0)

    def predict(self,x):
        doc_topic_distribution = self.get_doc_topic_distribution(x)
        print(doc_topic_distribution)
        topics = [np.where(p > 0.1) if len(np.where(p > 0.1)[0]) > 0 else np.where(p>=0.01) for p in doc_topic_distribution]
        return topics
