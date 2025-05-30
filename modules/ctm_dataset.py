import torch
from torch.utils.data import Dataset
import scipy.sparse

class CTMDataset(Dataset):

    """Class to load BoW and the contextualized embeddings."""

    def __init__(self, X_contextual, X_bow=None, idx2token=None, labels=None):

        if X_bow.shape[0] != len(X_contextual):
            raise Exception("Wait! BoW and Contextual Embeddings have different sizes! "
                            "You might want to check if the BoW preparation method has removed some documents. ")


        self.X_bow = X_bow
        self.X_contextual = X_contextual
        self.idx2token = idx2token
        self.labels = labels

    def __len__(self):
        """Return length of dataset."""
        return self.X_bow.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        if X_bow:
            X_bow = torch.FloatTensor(self.X_bow[i])
        X_contextual = torch.FloatTensor(self.X_contextual[i])

        return_dict = {'X_bow': X_bow, 'X_contextual': X_contextual}

        return return_dict