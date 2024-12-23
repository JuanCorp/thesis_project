import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
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



def prior(K, alpha):
    """
    Prior for the model.
    :K: number of categories
    :alpha: Hyper param of Dir
    :return: mean and variance tensors
    """
    # Approximate to normal distribution using Laplace approximation
    a = torch.Tensor(1, K).float().fill_(alpha)
    return a


class Dirichlet_VAE(nn.Module):
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
            nn.Dropout(p=0.2),
            nn.Linear(in_features=100, out_features=self.topic_size)
        )
        
        self.encoder_norm = nn.BatchNorm1d(num_features=self.topic_size, affine=False)


        self.beta = nn.Parameter(torch.Tensor(self.topic_size,self.input_size))
        nn.init.xavier_uniform_(self.beta)

        # decoder
        
        self.decoder_norm = nn.BatchNorm1d(num_features=self.input_size, affine=False)
        self.drop_theta = nn.Dropout(0.2)

        alpha_prior = nn.Parameter(prior(self.topic_size,0.3))
        self.alpha_prior = alpha_prior
        
        # save hyperparameters

    def forward(self, x):
        alpha = F.softplus(self.encoder_norm(self.encoder(x)))
        alpha = torch.max(torch.tensor(0.00001, device=alpha.device), alpha)
        theta = rsvi(alpha)
        theta = self.drop_theta(theta)
        word_dist = F.softmax(
                self.decoder_norm(torch.matmul(theta, self.beta)), dim=1
        )
        # word_dist: batch_size x input_size
        self.topic_word_matrix = word_dist

        return (
            self.alpha_prior,
            alpha,
            word_dist
        )

    def get_posterior(self, x):
        """Get posterior distribution."""
        # batch_size x n_components
        alpha = F.softplus(self.encoder_norm(self.encoder(x)))
        alpha = torch.max(torch.tensor(0.00001, device=alpha.device), alpha)
        

        return alpha
    

    def get_theta(self, x):
        with torch.no_grad():
            # batch_size x n_components
            alpha = self.get_posterior(x)
            # posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = rsvi(alpha)
            return theta


 
    

    def sample(self, alpha, n_samples: int = 20):
        with torch.no_grad():
            theta = rsvi(alpha)

            return theta
    


    