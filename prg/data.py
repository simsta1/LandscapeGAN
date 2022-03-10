import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import PIL
import os

class CustomDataset(Dataset):
    """
    Class defines custom Dataset as a workaround to the ImageFolder class, which 
    did not return correct amount of batch size.
    """

    def __init__(self, root_dir: str, transform: 'Compose' = None):
        """
        
        Params:
        ------------------
        root_dir: str
            Defines the path from where all images should be imported from
        
        transform: torch.utils.transforms.Compose
            Compose of different transforms applied during import of an image.
            
        all_images: list
            List of all images names in the root dir.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.all_images = [img for img in os.listdir(self.root_dir) if '.jpg' in img]

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.all_images[idx])
        image = PIL.Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image
    
def get_dataloader(root_dir: str, transforms: 'torch.utils.Compose', 
                   batch_size: int, workers: int):
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
    
    custom_dataset = CustomDataset(root_dir=root_dir, transform=transforms)
    
    return DataLoader(dataset=custom_dataset, 
                      batch_size=batch_size, 
                      num_workers=workers)
    
    
    
    
    