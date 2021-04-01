import os
import random
import re
from PIL import Image

DATA_PATH = "D:/eyeset/videos2/grayscale"
FRAME_PATH = DATA_PATH+"/train"
MASK_PATH = DATA_PATH+"/masks"

folders = ['/train_frames', '/train_masks', '/val_frames', '/val_masks', '/test_frames', '/test_masks']

for folder in folders:
    if not os.path.exists(DATA_PATH + folder):
        os.makedirs(DATA_PATH + folder)
  
all_frames = os.listdir(FRAME_PATH)
all_masks = os.listdir(MASK_PATH)

#just make sure the ids are actually in order
all_frames.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])
all_masks.sort(key=lambda var:[int(x) if x.isdigit() else x 
                               for x in re.findall(r'[^0-9]|[0-9]+', var)])

random.seed(187372311)
random.shuffle(all_frames)

train_split = int(0.7*len(all_frames))
val_split = int(0.9 * len(all_frames))

train_frames = all_frames[:train_split]
val_frames = all_frames[train_split:val_split]
test_frames = all_frames[val_split:]

train_masks = [f for f in all_masks if f in train_frames]
val_masks = [f for f in all_masks if f in val_frames]
test_masks = [f for f in all_masks if f in test_frames]

def add_frames(dir_name, image):
    img = Image.open(FRAME_PATH+'/'+image)
    img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)

def add_masks(dir_name, image):
    img = Image.open(MASK_PATH+'/'+image)
    img.save(DATA_PATH+'/{}'.format(dir_name)+'/'+image)

frame_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'), 
                 (test_frames, 'test_frames')]
mask_folders = [(train_masks, 'train_masks'), (val_masks, 'val_masks'), 
                (test_masks, 'test_masks')]

for folder in frame_folders:
    array = folder[0]
    name = [folder[1]] * len(array)
    list(map(add_frames, name, array))

for folder in mask_folders:
    array = folder[0]
    name = [folder[1]] * len(array)
    list(map(add_masks, name, array))