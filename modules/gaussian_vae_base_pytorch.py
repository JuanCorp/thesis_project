import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)


class Gaussian_VAE(nn.Module):
    def __init__(self,
                 input_size,
                 embedding_size,
                 topic_size,
                 beta=2.0):
        super().__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.topic_size = topic_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.embedding_size, out_features=100),
            nn.Softplus(),
            nn.Linear(in_features=100, out_features=100),
            nn.Softplus(),
            nn.Dropout(p=0.2)
        )
        self.f_mu = nn.Linear(100,self.topic_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(self.topic_size, affine=False)

        self.f_sigma = nn.Linear(100, self.topic_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(self.topic_size, affine=False)

        topic_prior_mean = 0.0
        self.prior_mean = torch.tensor([topic_prior_mean] * self.topic_size).cuda()
        self.prior_mean = nn.Parameter(self.prior_mean)
        topic_prior_variance = 1.0 - (1.0/self.topic_size)
        self.prior_variance = torch.tensor([topic_prior_variance] * self.topic_size).cuda()
        self.prior_variance = nn.Parameter(self.prior_variance)
        self.beta = nn.Parameter(torch.Tensor(self.topic_size,self.input_size))
        nn.init.xavier_uniform_(self.beta)

        # decoder
        
        self.decoder_norm = nn.BatchNorm1d(num_features=self.input_size, affine=False)
        self.drop_theta = nn.Dropout(0.2)

        # save hyperparameters

    def forward(self, x):
        x = self.encoder(x)
        posterior_mu = self.f_mu_batchnorm(self.f_mu(x))
        posterior_log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))
        posterior_sigma = torch.exp(posterior_log_sigma)
        theta = F.softmax(reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        theta = self.drop_theta(theta)
        word_dist = F.softmax(
                self.decoder_norm(torch.matmul(theta, self.beta)), dim=1
        )
        # word_dist: batch_size x input_size
        self.topic_word_matrix = self.beta

        return (
            self.prior_mean,
            self.prior_variance,
            posterior_mu,
            posterior_sigma,
            posterior_log_sigma,
            word_dist
        )

    def get_posterior(self, x):
        """Get posterior distribution."""
        # batch_size x n_components
        x = self.encoder(x)
        posterior_mu = self.f_mu_batchnorm(self.f_mu(x))
        posterior_log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return posterior_mu, posterior_log_sigma
    

    def get_theta(self, x):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.get_posterior(x)
            # posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                reparameterize(posterior_mu, posterior_log_sigma), dim=1
            )

            return theta

    def sample(self, posterior_mu, posterior_log_sigma, n_samples: int = 20):
        with torch.no_grad():
            posterior_mu = posterior_mu.unsqueeze(0).repeat(n_samples, 1, 1)
            posterior_log_sigma = posterior_log_sigma.unsqueeze(0).repeat(n_samples, 1, 1)
            # generate samples from theta
            theta = F.softmax(
                reparameterize(posterior_mu, posterior_log_sigma), dim=-1
            )

            return theta.mean(dim=0)
    


    