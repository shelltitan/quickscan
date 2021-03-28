from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
val_datagen = ImageDataGenerator(rescale=1./255)

train_image_generator = train_datagen.flow_from_directory(
'D:/eyeset/videos2/grayscale/train_frames',color_mode='grayscale',
batch_size = 8)#NORMALLY 4/8/16/32

train_mask_generator = train_datagen.flow_from_directory(
'D:/eyeset/videos2/grayscale/train_masks',color_mode='grayscale',
batch_size = 8)#NORMALLY 4/8/16/32)

val_image_generator = val_datagen.flow_from_directory(
'D:/eyeset/videos2/grayscale/val_frames',color_mode='grayscale',
batch_size = 4)#NORMALLY 4/8/16/32)


val_mask_generator = val_datagen.flow_from_directory(
'D:/eyeset/videos2/grayscale/val_masks',color_mode='grayscale',
batch_size = 4)#NORMALLY 4/8/16/32)

train_generator = tensorflow.tuple(train_image_generator, train_mask_generator)
val_generator = tensorflow.tuple(val_image_generator, val_mask_generator)
