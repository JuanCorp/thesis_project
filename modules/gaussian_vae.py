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


class DVAE(pl.LightningModule):
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
        self.save_hyperparameters()

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


    def training_step(self, batch, batch_idx):
        X_bow = batch["X_bow"]
        X_bow = X_bow.reshape(X_bow.shape[0], -1)
        X_contextual = batch["X_contextual"]
        prior_mean,prior_variance,posterior_mu,posterior_sigma,posterior_log_sigma,word_dist = self(X_contextual)
        kl, recon = self.objective(X_bow,prior_mean,prior_variance,posterior_mu,posterior_sigma,posterior_log_sigma,word_dist)
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
        X_bow = batch["X_bow"]
        X_bow = X_bow.reshape(X_bow.shape[0], -1)
        X_contextual = batch["X_contextual"]
        prior_mean,prior_variance,posterior_mu,posterior_sigma,posterior_log_sigma,word_dist = self(X_contextual)
        kl, recon = self.objective(X_bow,prior_mean,prior_variance,posterior_mu,posterior_sigma,posterior_log_sigma,word_dist)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-3,betas=(0.99,0.99))
        return optimizer
    

    def objective(
        self,
        inputs,
        prior_mean,
        prior_variance,
        posterior_mean,
        posterior_variance,
        posterior_log_variance,
        word_dists
    ):

        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = prior_variance.log().sum() - posterior_log_variance.sum(
            dim=1
        )
        # combine terms
        KL = 0.5 * (var_division + diff_term - self.topic_size + logvar_det_division)

        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)

        # loss = self.weights["beta"]*KL + RL
        return KL, RL
    

    def sample(self, posterior_mu, posterior_log_sigma, n_samples: int = 20):
        with torch.no_grad():
            posterior_mu = posterior_mu.unsqueeze(0).repeat(n_samples, 1, 1)
            posterior_log_sigma = posterior_log_sigma.unsqueeze(0).repeat(n_samples, 1, 1)
            # generate samples from theta
            theta = F.softmax(
                reparameterize(posterior_mu, posterior_log_sigma), dim=-1
            )

            return theta.mean(dim=0)
    


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
                X_contextual = batch_samples['X_contextual']

                # forward pass
                self.zero_grad()
                mu, log_var = self.get_posterior(X_contextual)
                thetas = self.sample(mu, log_var, n_samples).cpu().numpy()
                final_thetas.append(thetas)
        return np.concatenate(final_thetas, axis=0)
    

    def get_most_likely_topic(self, doc_topic_distribution):
        """get the most likely topic for each document

        :param doc_topic_distribution: ndarray representing the topic distribution of each document
        """
        return np.argmax(doc_topic_distribution, axis=1)
    
    def predict(self,dataset):
        doc_topic_distribution = self.get_doc_topic_distribution(dataset)
        print(doc_topic_distribution)
        topics = [np.where(p > 0.1) if len(np.where(p > 0.1)[0]) > 0 else np.where(p>=0.1) for p in doc_topic_distribution]
        return topics