import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import PIL
import os
<<<<<<< HEAD
import numpy as np
=======
>>>>>>> 47d48b6526b303a6acb21ca69d041332a4bf51f8

class CustomDataset(Dataset):
    """
    Class defines custom Dataset as a workaround to the ImageFolder class, which 
    did not return correct amount of batch size.
    """

<<<<<<< HEAD
    def __init__(self, root_dir: str, transform: 'Compose' = None, dataset_fraction: float = 1):
=======
    def __init__(self, root_dir: str, transform: 'Compose' = None):
>>>>>>> 47d48b6526b303a6acb21ca69d041332a4bf51f8
        """
        
        Params:
        ------------------
        root_dir: str
            Defines the path from where all images should be imported from
        
        transform: torch.utils.transforms.Compose
            Compose of different transforms applied during import of an image.
            
        all_images: list
            List of all images names in the root dir.
<<<<<<< HEAD
        
        dataset_fraction: int = 1
            fraction of dataset to take from for training. default = 1
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_fraction = dataset_fraction
        self.images = self._build_image_list()
        
    def _build_image_list(self):
        # Calculate fraction of data to take from dataset
        dataset_size = int(round(len(os.listdir(self.root_dir)) * self.dataset_fraction, 0))
        # Load all images from within root dri
        all_images = np.array([img for img in os.listdir(self.root_dir) if '.jpg' in img])
        
        return np.random.permutation(all_images)[:dataset_size]
    
    def get_len_root(self):
        return len(os.listdir(self.root_dir))
        
    def __len__(self):
        return len(self.images)
=======
        """
        self.root_dir = root_dir
        self.transform = transform
        self.all_images = [img for img in os.listdir(self.root_dir) if '.jpg' in img]

    def __len__(self):
        return len(os.listdir(self.root_dir))
>>>>>>> 47d48b6526b303a6acb21ca69d041332a4bf51f8

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
<<<<<<< HEAD
        img_name = os.path.join(self.root_dir, self.images[idx])
=======
        img_name = os.path.join(self.root_dir, self.all_images[idx])
>>>>>>> 47d48b6526b303a6acb21ca69d041332a4bf51f8
        image = PIL.Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
    
def get_dataloader(root_dir: str, transforms: 'torch.utils.Compose', 
<<<<<<< HEAD
                   batch_size: int, workers: int, dataset_fraction: float = 1):
=======
                   batch_size: int, workers: int):
>>>>>>> 47d48b6526b303a6acb21ca69d041332a4bf51f8
    """
    Functino returns a dataloader with given parameters
    
    Params:
    ---------------
    root_dir: str
        Defines the path from where all images should be imported from
        
    transform: torch.utils.transforms.Compose
        Compose of different transforms applied during import of an image.
        
    batch_size; int
        Size of the imported batch
        
    woerkers: int
        Amount of CPU workers for the loading of data into gpu.
    """
    
<<<<<<< HEAD
    custom_dataset = CustomDataset(root_dir=root_dir, transform=transforms, dataset_fraction=dataset_fraction)
=======
    custom_dataset = CustomDataset(root_dir=root_dir, transform=transforms)
>>>>>>> 47d48b6526b303a6acb21ca69d041332a4bf51f8
    
    return DataLoader(dataset=custom_dataset, 
                      batch_size=batch_size, 
                      num_workers=workers)
    
    
    
    
    