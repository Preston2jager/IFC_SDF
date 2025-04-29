import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time

import m02_Data_Files.d03_SDF_Converted

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SDFDataset(Dataset):
    """
    TODO: adapting to handle multiple objects
    """
    def __init__(self):
        start = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] SDFDataset init started")

        samples_dict = np.load(os.path.join(os.path.dirname(m02_Data_Files.d03_SDF_Converted.__file__), f'samples_dict.npy'), allow_pickle=True).item()
        self.data = dict()
        for obj_idx in list(samples_dict.keys()):  # samples_dict.keys() for all the objects
            for key in samples_dict[obj_idx].keys():   # keys are ['samples', 'sdf', 'latent_class', 'samples_latent_class']
                value = torch.from_numpy(samples_dict[obj_idx][key]).float()
                if len(value.shape) == 1:    # increase dim if monodimensional, needed to vstack
                    value = value.view(-1, 1)
                if key not in list(self.data.keys()):
                    self.data[key] = value
                else:
                    self.data[key] = torch.vstack((self.data[key], value))
        print(f"[{time.strftime('%H:%M:%S')}] SDFDataset init done, took {time.time() - start:.2f}s")
        return

    def __len__(self):
        return self.data['sdf'].shape[0]

    def __getitem__(self, idx):
        latent_class = self.data['samples_latent_class'][idx, :]
        sdf = self.data['sdf'][idx]
        return latent_class, sdf

if __name__=='__main__':
    dataset = SDFDataset()
