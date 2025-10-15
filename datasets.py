import numpy as np
import torch
import random
import os
from utils import StdScaler


class KolmogorovFlowDataset:
    def __init__(self, norm=True, shuffle=True, n_concurrent=3, seed=1234, size=-1, data_folder="data"):
        # Setting random seed for reproducibility
        torch.torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        filename = os.path.join(data_folder, "kf_2d_re1000_256_40seed.npy")
        self.raw_data = torch.Tensor(np.load(filename))
        self.scaler = StdScaler(0.0, 4.7852)

        self.nexp, self.ntimesteps, self.ny, self.nx = self.raw_data.shape
        self.norm = norm
        self.n_concurrent = n_concurrent

        samples_xexp = self.ntimesteps // n_concurrent
        samples_extra = self.ntimesteps % n_concurrent

        if samples_extra > 0: 
            self.raw_data = self.raw_data[:, :-samples_extra]

        self.data = self.raw_data.reshape(self.nexp, samples_xexp, n_concurrent, self.ny, self.nx)
        self.data = self.data.reshape(self.nexp * samples_xexp, n_concurrent, self.ny, self.nx)

        if shuffle:
            indices = torch.randperm(self.data.size(0))
            self.data = self.data[indices]

        if norm:
            self.data = self.scaler(self.data)

        self.size = self.data.shape[0] if size < 0 else size
        self.data = self.data[:self.size]


    def __len__(self):
        return self.size
    

    def __getitem__(self, idx):
        return self.data[idx].clone()


    def to_numpy(self):
        return self.data.numpy()
    

    def split(self, perc=0.75):
        # Return train and test sets
        N = int(self.__len__() * perc)
        return self.data[:N].clone(), self.data[N:].clone()
