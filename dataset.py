import os
import random
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile

# code from: https://github.com/AntixK/PyTorch-VAE/blob/master/dataset.py
# normalization: https://github.com/chingyaoc/fair-mixup/blob/master/celeba/utils.py

class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    #def __init__(self, root, split, target_type, transform, target_transform, download) -> None:
    #    super().__init__(root, split, target_type, transform, target_transform, download) 
    #    
    #    # protected attribute
    #    # https://www.kaggle.com/nageshsingh/gender-detection-using-inceptionv3-92-6-acc
    #    self.protected_attribute = self.attr[:, 20] # gender is index 20
    #  
    #    print(self.protected_attribute.shape)
    #    assert False

    def _check_integrity(self) -> bool:
        return True

#def transform_CelebA():
#    return transforms.Compose([transforms.RandomHorizontalFlip(),
#                                              transforms.CenterCrop(148),
#                                              transforms.Resize(self.patch_size),
#                                              transforms.ToTensor(),
#                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html

def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)

class ConditionalBatchSampler(Sampler): # only sample from either group 0 or group 1 in a batch, not both
    def __init__(self, dataset, batch_size, protected_attribute):
        protected_attribute = protected_attribute.int()
        assert len(torch.unique(protected_attribute)) == 2 # assert binary attribute
        self.first_group_indices = [i for i in range(len(protected_attribute)) if protected_attribute[i] == 0]
        self.second_group_indices = [i for i in range(len(protected_attribute)) if protected_attribute[i] == 1]
        self.batch_size = batch_size

    def __iter__(self):
        random.shuffle(self.first_group_indices)
        random.shuffle(self.second_group_indices)
        first_group_batches  = chunk(self.first_group_indices, self.batch_size)
        second_group_batches = chunk(self.second_group_indices, self.batch_size)
        combined = list(first_group_batches + second_group_batches)
        combined = [batch.tolist() for batch in combined]
        random.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return (len(self.first_group_indices) + len(self.second_group_indices)) // self.batch_size
