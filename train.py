from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger ,EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import os

import modelVNET
import jaccard_coef_loss
import data_augmentation

NO_OF_TRAINING_IMAGES = len(os.listdir('D:/eyeset/videos2/grayscale/train_frames/train'))
NO_OF_VAL_IMAGES = len(os.listdir('D:/eyeset/videos2/grayscale/val_frames/val'))

NO_OF_EPOCHS = 100
BATCH_SIZE = 8

weights_path = 'weights'

model = modelVNET.VNet()
opt = Adam(lr=1E-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss=jaccard_coef_loss.jaccard_distance_loss,
              optimizer=opt,
              metrics='acc')

checkpoint = ModelCheckpoint(weights_path, monitor=jaccard_coef_loss.jaccard_distance_loss, 
                             verbose=1, save_best_only=True, mode='max')

csv_logger = CSVLogger('./log.out', append=True, separator=';')

earlystopping = EarlyStopping(monitor = jaccard_coef_loss.jaccard_distance_loss, verbose = 1,
                              min_delta = 0.01, patience = 3, mode = 'max')

lr_callback = ReduceLROnPlateau(min_lr=0.000001)

callbacks_list = [checkpoint, csv_logger, earlystopping]


results = model.fit_generator(data_augmentation.train_generator,
                              epochs=NO_OF_EPOCHS, 
                              steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                              validation_data = data_augmentation.val_generator, 
                              validation_steps = (NO_OF_VAL_IMAGES//4),
                              verbose=1,
                              shuffle=True,
                              callbacks=callbacks_list)

model.save("models/"+"VNet.h5")
