import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence
from torch.utils.data import DataLoader, TensorDataset

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, latent_dim)  # alpha parameters for Dirichlet

    def forward(self, x):
        h = F.relu(self.fc1(x))
        alpha = F.softplus(self.fc2(h)) + 1e-6  # Ensure alpha > 0
        return alpha


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        reconstruction = torch.sigmoid(self.fc2(h))
        return reconstruction


class DVAE(pl.LightningModule):
    def __init__(self,input_size, embedding_size, topic_size, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = Encoder(embedding_size, topic_size)
        self.decoder = Decoder(topic_size, input_size)
        self.latent_dim = topic_size
        self.learning_rate = learning_rate

    def reparameterize(self, alpha):
        """Sample from a Dirichlet distribution with the given alpha."""
        dist = Dirichlet(alpha)
        return dist.rsample(), dist

    def forward(self, x):
        alpha = self.encoder(x)
        z, dist = self.reparameterize(alpha)
        reconstruction = self.decoder(z)
        return reconstruction, alpha, z, dist

    def objective(self, x, reconstruction, alpha, dist):
        """Calculate the VAE loss: Reconstruction + KL divergence."""
        recon_loss = -torch.sum(x * torch.log(reconstruction + 1e-10), dim=1)

        # KL divergence between Dirichlet and a uniform Dirichlet prior
        prior = Dirichlet(torch.ones_like(alpha) * 0.02)
        kl_loss = kl_divergence(dist, prior).sum()

        return recon_loss + kl_loss

    def training_step(self, batch, batch_idx):
        X_bow = batch["X_bow"]
        X_bow = X_bow.reshape(X_bow.shape[0], -1)
        X_contextual = batch["X_contextual"]
        reconstruction, alpha, z, dist = self.forward(X_contextual)
        loss = self.objective(X_bow, reconstruction, alpha, dist)
        loss = loss.sum()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_bow = batch["X_bow"]
        X_bow = X_bow.reshape(X_bow.shape[0], -1)
        X_contextual = batch["X_contextual"]
        reconstruction, alpha, z, dist = self.forward(X_contextual)
        loss = self.objective(X_bow, reconstruction, alpha, dist)
        loss = loss.sum()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def predict(self,x):
        reconstruction, alpha, z, dist = self.forward(x)
        keep = torch.argmax(z,dim=1)
        assigned_topics = [[a] for a in keep]
        return assigned_topics