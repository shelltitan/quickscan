import torch
import albumentations
from models.unet import UNet

model = UNet(in_channels=1,
             out_channels=2,
             n_blocks=4,
             start_filters=32,
             activation='relu',
             normalization='batch',
             conv_mode='same',
             dim=2)

x = torch.randn(size=(1, 1, 720, 480), dtype=torch.float32)
with torch.no_grad():
    out = model(x)

print(f'Out: {out.shape}')