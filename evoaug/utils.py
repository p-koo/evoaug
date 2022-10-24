import os, pathlib, h5py
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader


class H5DataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=128):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        # Assign train and val split(s) for use in DataLoaders
        if stage == "fit" or stage is None:
            with h5py.File(self.data_path, 'r') as dataset:
                self.x_train = torch.from_numpy(np.array(dataset["X_train"]).astype(np.float32))
                self.y_train = torch.from_numpy(np.array(dataset["Y_train"]).astype(np.float32))
                self.x_valid = torch.from_numpy(np.array(dataset["X_valid"]).astype(np.float32))
                self.y_valid = torch.from_numpy(np.array(dataset["Y_valid"]).astype(np.float32))
            _, self.A, self.L = self.x_train.shape # N = number of seqs, A = alphabet size (number of nucl.), L = length of seqs
            self.num_classes = self.y_train.shape[1]
            
        # Assign test split(s) for use in DataLoaders
        if stage == "test" or stage is None:
            with h5py.File(self.data_path, "r") as dataset:
                self.x_test = torch.from_numpy(np.array(dataset["X_test"]).astype(np.float32))
                self.y_test = torch.from_numpy(np.array(dataset["Y_test"]).astype(np.float32))
            _, self.A, self.L = self.x_train.shape
            self.num_classes = self.y_train.shape[1]
            
    def train_dataloader(self):
        train_dataset = TensorDataset(self.x_train, self.y_train) # tensors are index-matched
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) # sets of (x, x', y) will be shuffled
    
    def val_dataloader(self):
        valid_dataset = TensorDataset(self.x_valid, self.y_valid) 
        return DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        test_dataset = TensorDataset(self.x_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) 



def make_directory(directory):
    """make directory"""
    if not os.path.isdir(directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        print("Making directory: " + directory)




