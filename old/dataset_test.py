import dataset
from torch.utils import data

images = ['D:/eyeset/videos2/grayscale/train_frames/frame (1).png', 'D:/eyeset/videos2/grayscale/train_frames/frame (2).png']
masks = ['D:/eyeset/videos2/grayscale/train_masks/frame (1).png', 'D:/eyeset/videos2/grayscale/train_masks/frame (2).png']

training_dataset = dataset.EyeDataset(images=images,
                              masks=masks,
                              transform=None)

training_dataloader = data.DataLoader(dataset=training_dataset,
                                      batch_size=2,
                                      shuffle=True)
x, y = next(iter(training_dataloader))

print(f'x = shape: {x.shape}; type: {x.dtype}')
print(f'x = min: {x.min()}; max: {x.max()}')
print(f'y = shape: {y.shape}; class: {y.unique()}; type: {y.dtype}')