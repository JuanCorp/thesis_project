
import matplotlib.pyplot as plt
import numpy as np
import torch
import wordcloud
from scipy.special import softmax
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextualized_topic_models.utils.early_stopping.early_stopping import (
    EarlyStopping,
)
from modules.dir_vae_base_pytorch import Dirichlet_VAE
import datetime
from torch.distributions import Dirichlet


class DIR_VAE:
    def __init__(
        self,
        bow_size,
        contextual_size,
        n_components=20,
        dropout=0.2,
        learn_priors=True,
        batch_size=64,
        lr=2e-2,
        momentum=0.99,
        solver="adam",
        num_epochs=50,
        beta=1,
        prior=None
    ):

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )


        self.bow_size = bow_size
        self.n_components = n_components
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.contextual_size = contextual_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.training_doc_topic_distributions = None
        self.beta = beta
        self.prior=prior
        self.val_kl_losses = list()

        self.model = Dirichlet_VAE(
            bow_size,
            self.contextual_size,
            n_components
        )

        self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99)
            )

        # performance attributes
        self.best_loss_train = float("inf")

        # training attributes
        self.model_dir = None
        self.nn_epoch = None

        # validation attributes
        self.validation_data = None

        # learned topics
        self.best_components = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        self.model = self.model.to(self.device)


    def _loss(
        self,
        inputs,
        prior_alpha,
        posterior_alpha,
        word_dists
    ):
        prior = Dirichlet(prior_alpha)
        posterior = Dirichlet(posterior_alpha)
        KL = torch.distributions.kl.kl_divergence(posterior, prior)
        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        # loss = self.weights["beta"]*KL + RL
        return KL, RL
    

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in loader:
            # batch_size x vocab_size
            X_bow = batch_samples["X_bow"]
            if len(X_bow.shape) > 2:
                X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_samples["X_contextual"]
            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # forward pass
            self.model.zero_grad()
            (
                prior_alpha,
                posterior_alpha,
                word_dists
            ) = self.model(X_contextual)
            
            if self.prior is not None:
                prior_alpha =self.prior.cuda()
            # backward pass
            kl_loss, rl_loss = self._loss(
                X_bow,
                prior_alpha,
                posterior_alpha,
                word_dists
            )

            loss = self.beta * kl_loss + rl_loss
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X_bow.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss
    
    def _validation(self, loader):
        """Validation epoch."""
        self.model.eval()
        val_loss = 0
        epoch_kl_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            # batch_size x vocab_size
            X_bow = batch_samples["X_bow"]
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_samples["X_contextual"]

            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # forward pass
            self.model.zero_grad()
            (
                prior_alpha,
                posterior_alpha,
                word_dists
            ) = self.model(X_contextual)

            # backward pass
            kl_loss, rl_loss = self._loss(
                X_bow,
                prior_alpha,
                posterior_alpha,
                word_dists
            )
            epoch_kl_loss += (kl_loss)
            loss = self.beta * kl_loss + rl_loss
            loss = loss.sum()


            # compute train loss
            samples_processed += X_bow.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed
        epoch_kl_loss /= samples_processed        
        self.val_kl_losses.append(epoch_kl_loss.mean().cpu().detach().numpy())
        return samples_processed, val_loss


    def fit(
        self,
        train_dataset,
        validation_dataset=None,
        save_dir=None,
        verbose=False,
        patience=5,
        delta=0,
        n_samples=20,
        do_train_predictions=True,
    ):
        """
        Train the CTM model.

        :param train_dataset: PyTorch Dataset class for training data.
        :param validation_dataset: PyTorch Dataset class for validation data. If not None, the training stops if validation loss doesn't improve after a given patience
        :param save_dir: directory to save checkpoint models to.
        :param verbose: verbose
        :param patience: How long to wait after last time validation loss improved. Default: 5
        :param delta: Minimum change in the monitored quantity to qualify as an improvement. Default: 0
        :param n_samples: int, number of samples of the document topic distribution (default: 20)
        :param do_train_predictions: bool, whether to compute train predictions after fitting (default: True)
        """
        # Print settings to output file
        if verbose:
            print(
                "Settings: \n\
                   N Components: {}\n\
                   Topic Prior Mean: {}\n\
                   Topic Prior Variance: {}\n\
                   Model Type: {}\n\
                   Hidden Sizes: {}\n\
                   Activation: {}\n\
                   Dropout: {}\n\
                   Learn Priors: {}\n\
                   Learning Rate: {}\n\
                   Momentum: {}\n\
                   Reduce On Plateau: {}\n\
                   Save Dir: {}".format(
                    self.n_components,
                    0.0,
                    1.0 - (1.0 / self.n_components),
                    self.dropout,
                    self.learn_priors,
                    self.lr,
                    self.momentum,
                    save_dir,
                )
            )

        self.model_dir = save_dir
        self.idx2token = train_dataset.idx2token
        train_data = train_dataset
        self.validation_data = validation_dataset
        if self.validation_data is not None:
            self.early_stopping = EarlyStopping(
                patience=patience, verbose=verbose, path=save_dir, delta=delta
            )
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

        # init training variables
        samples_processed = 0

        # train loop
        pbar = tqdm(self.num_epochs, position=0, leave=True)
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss = self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()
            pbar.update(1)

            if self.validation_data is not None:
                validation_loader = DataLoader(
                    self.validation_data,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validation(validation_loader)
                e = datetime.datetime.now()

                # report
                if verbose:
                    print(
                        "Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                            epoch + 1,
                            self.num_epochs,
                            val_samples_processed,
                            len(self.validation_data) * self.num_epochs,
                            val_loss,
                            e - s,
                        )
                    )

                pbar.set_description(
                    "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tValid Loss: {}\tTime: {}".format(
                        epoch + 1,
                        self.num_epochs,
                        samples_processed,
                        len(train_data) * self.num_epochs,
                        train_loss,
                        val_loss,
                        e - s,
                    )
                )

                self.early_stopping(val_loss, self)
                if self.early_stopping.early_stop:
                    print("Early stopping")

                    break
            else:
                # save last epoch
                self.best_components = self.model.beta
                if save_dir is not None:
                    self.save(save_dir)
            pbar.set_description(
                "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                    epoch + 1,
                    self.num_epochs,
                    samples_processed,
                    len(train_data) * self.num_epochs,
                    train_loss,
                    e - s,
                )
            )

        pbar.close()
        if do_train_predictions:
            self.training_doc_topic_distributions = self.get_doc_topic_distribution(
                train_dataset, n_samples
            )

    
    def save(self,dir):
        #Don't do nuthin
        pass

    def get_doc_topic_distribution(self, dataset, n_samples=20):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of sample to collect to estimate the final distribution (the more the better).
        """
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False
        )
        final_thetas = []
        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X_contextual = batch_samples['X_contextual'].cuda()

                # forward pass
                self.model.zero_grad()
                mu = self.model.get_posterior(X_contextual)
                thetas = self.model.sample(mu, n_samples).cpu().numpy()
                final_thetas.append(thetas)
        return np.concatenate(final_thetas, axis=0)
    

    def get_most_likely_topic(self, doc_topic_distribution):
        """get the most likely topic for each document

        :param doc_topic_distribution: ndarray representing the topic distribution of each document
        """
        return np.argmax(doc_topic_distribution, axis=1)
    
    def predict(self,dataset):
        doc_topic_distribution = self.get_doc_topic_distribution(dataset)
        topics = [np.where(p > 0.1) if len(np.where(p > 0.1)[0]) > 0 else np.where(p>=0.01) for p in doc_topic_distribution]
        return topics
    
    def get_posterior(self,dataset):
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False
        )
        final_thetas = []
        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X_contextual = batch_samples['X_contextual'].cuda()

                # forward pass
                self.model.zero_grad()
                mu = self.model.get_posterior(X_contextual)
                final_thetas.append(mu)
        return torch.cat(final_thetas, dim=0)
    

    def get_word_topic_matrix(self,dataset):
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=64,
            shuffle=False
        )
        final_thetas = []
        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X_contextual = batch_samples['X_contextual'].cuda()

                # forward pass
                self.model.zero_grad()
                _,_,mu = self.model(X_contextual)
                final_thetas.append(mu)
        return torch.cat(final_thetas, dim=0).cpu().numpy()
    

    
    def get_top_tokens(self,idx2word):
        topics = self.model.beta.detach().cpu().numpy()
        print(topics.shape)
        topics = topics.argsort(axis=1)[:, ::-1]
        # top 10 words
        topics = topics[:, :10]
        print(topics.shape)
        print(topics)
        top_tokens = [[idx2word[i] for i in topic] for topic in topics]
        print(len(top_tokens))
        return top_tokens
