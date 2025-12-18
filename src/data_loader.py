import torch 
import numpy as np 
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Dataloader:
    def __init__(self, data_path, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size

        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        

    def get_batch(self, split='train'):
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))

        x = torch.stack([torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+self.block_size+1]).astype(np.int64)) for i in ix])
        
        return x.to(device), y.to(device)