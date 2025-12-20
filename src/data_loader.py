import torch 
import numpy as np 
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataloader:
    def __init__(self, data_path, batch_size, block_size, split_ratio=0.9):
        self.batch_size = batch_size
        self.block_size = block_size
        
        # Load the binary data
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # Split into train and validation sets
        n = int(split_ratio * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        
    def get_batch(self, split, device):
        # Select the correct data set
        data = self.train_data if split == 'train' else self.val_data
        
        # Generate random starting indices
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        
        # Pull chunks
        x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+self.block_size+1]).astype(np.int64)) for i in ix])
        
        return x.to(device), y.to(device)