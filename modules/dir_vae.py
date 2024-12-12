import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
import pytorch_lightning as pl

def prior(K, alpha):
    """
    Prior for the model.
    :K: number of categories
    :alpha: Hyper param of Dir
    :return: mean and variance tensors
    """
    # Approximate to normal distribution using Laplace approximation
    a = torch.Tensor(1, K).float().fill_(alpha)
    mean = a.log().t() - a.log().mean(1)
    var = ((1 - 2.0 / K) * a.reciprocal()).t() + (1.0 / K ** 2) * a.reciprocal().sum(1)
    return mean.t(), var.t() # Parameters of prior distribution after approximation



class DVAE(pl.LightningModule):
    def __init__(self,
                 embedding_size,
                 topic_size,
                 beta=2.0):
        super().__init__()

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

        )
        #CTM data set+
        #LayerNorm
        self.encoder_mean = nn.Linear(in_features=100, out_features=self.topic_size)
        self.encoder_var = nn.Linear(in_features=100, out_features=self.topic_size)
        self.normalize = nn.BatchNorm1d(num_features=self.topic_size)
        

        # decoder
        self.decoder = nn.Linear(in_features=self.topic_size,out_features=self.embedding_size)
        self.decoder_norm = nn.BatchNorm1d(num_features=self.embedding_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_norm.weight.data.copy_(torch.ones(self.embedding_size))
        self.decoder_norm.weight.requires_grad = False


        # Dir prior
        self.prior_mean, self.prior_var = map(nn.Parameter, prior(self.topic_size, 0.3)) # 0.3 is a hyper param of Dirichlet distribution
        self.prior_logvar = nn.Parameter(self.prior_var.log())
        self.prior_mean.requires_grad = False
        self.prior_var.requires_grad = False
        self.prior_logvar.requires_grad = False
        # save hyperparameters
        self.save_hyperparameters()

    def encode(self,x):
        encoded = self.encoder(x)
        return self.normalize(self.encoder_mean(encoded)),self.normalize(self.encoder_var(encoded))


    def decode(self,gauss_z):
        dir_z = F.softmax(gauss_z,dim=1)
        return F.sigmoid(self.decoder_norm(self.decoder(dir_z)))
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


    def forward(self, x):
        mu, logvar = self.encode(x)
        gauss_z = self.reparameterize(mu, logvar) 
        # gause_z is a variable that follows a multivariate normal distribution
        # Inputting gause_z into softmax func yields a random variable that follows a Dirichlet distribution (Softmax func are used in decoder)
        dir_z = F.softmax(gauss_z,dim=1) # This variable follows a Dirichlet distribution
        return self.decode(gauss_z), mu, logvar, gauss_z, dir_z

    def training_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, logvar, gauss_z, dir_z = self(x)
        recon, kl = self.objective(x_recon, x, mu, logvar, self.topic_size)
        loss = (recon + kl).mean()
        self.log_dict({'train/loss': loss,
                       'train/recon': recon,
                       'train/kl': kl.mean()},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon, mu, logvar, gauss_z, dir_z = self(x)
        recon, kl = self.objective(x_recon, x, mu, logvar, self.topic_size)
        loss = (recon + kl).mean()
        self.log_dict({'val/loss': loss,
                       'val/recon': recon,
                       'val/kl': kl.mean()},
                      prog_bar=True,
                      logger=True,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    

    def objective(self, recon_x, x, mu, logvar, K):
        beta = 1.0
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        # ディリクレ事前分布と変分事後分布とのKLを計算
        # Calculating KL with Dirichlet prior and variational posterior distributions
        # Original paper:"Autoencodeing variational inference for topic model"-https://arxiv.org/pdf/1703.01488
        prior_mean = self.prior_mean.expand_as(mu)
        prior_var = self.prior_var.expand_as(logvar)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        var_division = logvar.exp() / prior_var # Σ_0 / Σ_1
        diff = mu - prior_mean # μ_１ - μ_0
        diff_term = diff *diff / prior_var # (μ_1 - μ_0)(μ_1 - μ_0)/Σ_1
        logvar_division = prior_logvar - logvar # log|Σ_1| - log|Σ_0| = log(|Σ_1|/|Σ_2|)
        # KL
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(1) - K)        
        return BCE,KLD
    

    def predict(self,x):
        x_recon, mu, logvar, gauss_z, dir_z = self(x)
        z = dir_z
        keep = torch.argmax(z,dim=1)
        assigned_topics = [[a] for a in keep]
        return assigned_topics
