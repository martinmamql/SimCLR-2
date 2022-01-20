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


# CelebA Dataset
# code from: https://github.com/AntixK/PyTorch-VAE/blob/master/dataset.py
# normalization: https://github.com/chingyaoc/fair-mixup/blob/master/celeba/utils.py
class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    def _check_integrity(self) -> bool:
        return True

# UTK Dataset
# https://github.com/ArminBaz/UTK-Face
class UTKDataset(Dataset):
    '''
        Inputs:
            dataFrame : Pandas dataFrame
            transform : The transform to apply to the dataset
    '''
    def __init__(self, dataFrame, transform=None):
        # read in the transforms
        self.transform = transform
        
        # Use the dataFrame to get the pixel values
        data_holder = dataFrame.pixels.apply(lambda x: np.array(x.split(" "),dtype=float))
        arr = np.stack(data_holder)
        arr = arr / 255.0
        arr = arr.astype('float32')
        arr = arr.reshape(arr.shape[0], 48, 48, 1)
        # reshape into 48x48x1
        self.data = arr
        
        # get the age, gender, and ethnicity label arrays
        self.age_label = np.array(dataFrame.bins[:])         # Note : Changed dataFrame.age to dataFrame.bins
        self.gender_label = np.array(dataFrame.gender[:])
        self.eth_label = np.array(dataFrame.ethnicity[:])
    
    # override the length function
    def __len__(self):
        return len(self.data)
    
    # override the getitem function
    def __getitem__(self, index):
        # load the data at index and apply transform
        data = self.data[index]
        data = self.transform(data)
        
        # load the labels into a list and convert to tensors
        labels = torch.tensor((self.age_label[index], self.gender_label[index], self.eth_label[index]))
        
        # return data labels
        return data, labels

    def read_data():
        # Read in the dataframe
        dataFrame = pd.read_csv('../data/age_gender.gz', compression='gzip')
    
        # Construct age bins
        age_bins = [0,10,15,20,25,30,40,50,60,120]
        age_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        dataFrame['bins'] = pd.cut(dataFrame.age, bins=age_bins, labels=age_labels)
    
        # Split into training and testing
        train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.2)
    
        # get the number of unique classes for each group
        class_nums = {'age_num':len(dataFrame['bins'].unique()), 'eth_num':len(dataFrame['ethnicity'].unique()),
                      'gen_num':len(dataFrame['gender'].unique())}
    
        # Define train and test transforms
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49,), (0.23,))
        ])
    
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49,), (0.23,))
        ])
    
        # Construct the custom pytorch datasets
        train_set = UTKDataset(train_dataFrame, transform=train_transform)
        test_set = UTKDataset(test_dataFrame, transform=test_transform)
    
        # Load the datasets into dataloaders
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    
        # Sanity Check
        for X, y in train_loader:
            print(f'Shape of training X: {X.shape}')
            print(f'Shape of y: {y.shape}')
            break
    
        return train_loader, test_loader, class_nums



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

# For CelebA
#def transform_CelebA():
#    return transforms.Compose([transforms.RandomHorizontalFlip(),
#                                              transforms.CenterCrop(148),
#                                              transforms.Resize(self.patch_size),
#                                              transforms.ToTensor(),
#                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
