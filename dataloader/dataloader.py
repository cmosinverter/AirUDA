import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

import os, sys
import numpy as np
import random


class Load_Dataset(Dataset):
    def __init__(self, dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        # Load samples
        x_data = dataset["samples"]

        # Load labels
        y_data = dataset.get("labels")
        if y_data is not None and isinstance(y_data, np.ndarray):
            y_data = torch.from_numpy(y_data)
        
        # Convert to torch tensor
        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)
        
            
        self.x_data = x_data.float()
        self.y_data = y_data.float() if y_data is not None else None
        self.len = x_data.shape[0]
         

    def __getitem__(self, index):
        x = self.x_data[index]
        y = self.y_data[index] if self.y_data is not None else None
        return x, y

    def __len__(self):
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # loading dataset file from path
    dataset_file = torch.load(os.path.join(data_path, f"{domain_id}.pt"))

    # Loading datasets
    dataset = Load_Dataset(dataset_file, dataset_configs)



    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=hparams["batch_size"],
                                              num_workers=0)

    return data_loader

