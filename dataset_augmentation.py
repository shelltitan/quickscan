import torch
from torch.utils.data.dataset import Dataset
from skimage.io import imread
from torch.utils import data

torch.manual_seed(2953)

class EyeDataset(Dataset):
    def __init__(self,
                 images: list,
                 masks: list,
                 transform=None,
                 train=True):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.image_dtype = torch.float32
        self.mask_dtype = torch.long
        
    def __getitem__(self, index: int):
        # Select the sample
        image_ID = self.images[index]
        mask_ID = self.masks[index]
        
        # Load input and target
        x, y = imread(image_ID), imread(mask_ID)
        
        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)

        return x, y

    def __len__(self):
        return len(self.image_paths)