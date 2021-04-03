import torch
import torch.nn as nn
import albumentations
import pathlib
from dataset import EyeDataset
from models.VNet_model import VNet_Torch
from trainer import Trainer
from torch import optim
from transformations import Compose, AlbuSeg2d, DenseTarget, FixGreyScale
from transformations import MoveAxis, Normalize_to_01, Resize
from torch.utils.data import DataLoader
# root directory
root = pathlib.Path.cwd() / 'Data'
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
    DenseTarget(),
    AlbuSeg2d(albu=albumentations.HorizontalFlip(p=0.5)),
    AlbuSeg2d(albu=albumentations.Rotate(limit=360,p=0.2)),
    Normalize_to_01(),
    FixGreyScale()
])
# validation transformations
transforms_validation = Compose([
    DenseTarget(),
    Normalize_to_01(),
    FixGreyScale()
])

# dataset training
dataset_train = EyeDataset(inputs=images_train,
                           targets=masks_train,
                           transform=transforms)

# dataset validation
dataset_valid = EyeDataset(inputs=images_valid,
                           targets=masks_valid,
                           transform=transforms)

# dataloader training
dataloader_training = DataLoader(dataset=dataset_train,
                                 batch_size=8,
                                 shuffle=True)

# dataloader validation
dataloader_validation = DataLoader(dataset=dataset_valid,
                                   batch_size=4,
                                   shuffle=True)

# device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    torch.device('cpu')

# model
model = VNet_Torch().to(device)

# criterion
criterion = torch.nn.BCEWithLogitsLoss()

# optimizer
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, weight_decay=1e-8, momentum=0.9)

#learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

# trainer
trainer = Trainer(model=model,
                  device=device,
                  criterion=criterion,
                  optimizer=optimizer,
                  training_DataLoader=dataloader_training,
                  validation_DataLoader=dataloader_validation,
                  lr_scheduler=scheduler,
                  epochs=10,
                  epoch=0,
                  notebook=False)

# start training
training_losses, validation_losses, lr_rates = trainer.run_trainer()