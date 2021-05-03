import pathlib
import dataset
import albumentations
from transformations import Compose, Resize, DenseTarget, AlbuSeg2d, FixGreyScale
from transformations import MoveAxis, Normalize_to_01
from torch.utils.data import DataLoader

# root directory
root = pathlib.Path('D:/eyeset/videos2/grayscale')

def get_filenames_of_path(path: pathlib.Path, ext: str = '*'):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    return filenames

# input and target files
images_train = get_filenames_of_path(root / 'train_frames')
masks_train = get_filenames_of_path(root / 'train_masks')
images_valid = get_filenames_of_path(root / 'val_frames')
masks_valid = get_filenames_of_path(root / 'val_masks')

# training transformations and augmentations
transforms = Compose([
    Resize(input_size=(480, 720), target_size=(240, 360)),
    AlbuSeg2d(albu=albumentations.HorizontalFlip(p=0.5)),
    AlbuSeg2d(albu=albumentations.Rotate(limit=20,p=0.2)),
    DenseTarget(),
    Normalize_to_01(),
    FixGreyScale()
])

# dataset training
dataset_train = dataset.EyeDataset(inputs=images_train,
                                    targets=masks_train,
                                    transform=transforms)

# dataset validation
dataset_valid = dataset.EyeDataset(inputs=images_valid,
                                    targets=masks_valid,
                                    transform=transforms)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=2,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=2,
                                   shuffle=True)