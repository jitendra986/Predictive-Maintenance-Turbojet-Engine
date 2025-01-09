import numpy as np
from torch.utils.data import DataLoader, Dataset

class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, data, dummy_var, seq_length):
        self.data = data
        self.dummy_var = dummy_var
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Repeat dummy_var along the sequence length axis
        dummy_repeated = np.tile(self.dummy_var[index], (self.seq_length, 1))
        return self.data[index], dummy_repeated
