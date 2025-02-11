import torch
from torch.utils.data import Dataset
import scipy.sparse

class CTMDataset(Dataset):

    """Class to load BoW and the contextualized embeddings."""

    def __init__(self, X_contextual, X_bow=None, idx2token=None):


        self.X_bow = X_bow
        self.X_contextual = X_contextual
        self.idx2token = idx2token

    def __len__(self):
        """Return length of dataset."""
        return self.X_contextual.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        if self.X_bow is not None:
            X_bow = torch.FloatTensor(self.X_bow[i])
        else:
            X_bow = torch.FloatTensor(self.X_contextual[i])
        X_contextual = torch.FloatTensor(self.X_contextual[i])

        return_dict = {'X_bow': X_bow, 'X_contextual': X_contextual}

        return return_dict