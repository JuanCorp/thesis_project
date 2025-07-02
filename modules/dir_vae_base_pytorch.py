import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np



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
                 beta=2.0,
                 pretrained_encoder=None):
        super().__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.topic_size = topic_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_features=self.embedding_size, out_features=100),
            nn.Softplus(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=100, out_features=100),
            nn.Softplus(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=100, out_features=self.topic_size)
        )
        if pretrained_encoder is not None:
            self.encoder.load_state_dict(pretrained_encoder.state_dict())
        
        self.encoder_norm = nn.BatchNorm1d(num_features=self.topic_size, affine=False)


        self.beta = nn.Parameter(torch.Tensor(self.topic_size,self.input_size))
        nn.init.xavier_uniform_(self.beta)

        # decoder
        
        self.decoder_norm = nn.BatchNorm1d(num_features=self.input_size, affine=False)
        self.drop_theta = nn.Dropout(0.2)

        alpha_prior = nn.Parameter(prior(self.topic_size,0.002))
        self.alpha_prior = alpha_prior
        
        # save hyperparameters

    def forward(self, x):
        alpha = F.softplus(self.encoder_norm(self.encoder(x)))
        alpha = torch.max(torch.tensor(0.01, device=alpha.device), alpha)
        dist = Dirichlet(alpha)
        theta = dist.rsample()
        theta = self.drop_theta(theta)
        word_dist = F.softmax(
                self.decoder_norm(torch.matmul(theta, self.beta)), dim=1
        )  # beta + plus teacher weight
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
        alpha = torch.max(torch.tensor(0.01, device=alpha.device), alpha)
        

        return alpha
    

    def get_theta(self, x):
        with torch.no_grad():
            # batch_size x n_components
            alpha = self.get_posterior(x)
            # posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            dist = Dirichlet(alpha)
            theta = dist.rsample()
            return theta


 
    

    def sample(self, alpha, n_samples: int = 20):
        with torch.no_grad():
            dist = Dirichlet(alpha)
            theta = dist.rsample()

            return theta
    


    