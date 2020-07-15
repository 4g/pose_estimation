import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm

sm.set_framework('tf.keras')

from . import utils
from .callbacks import DisplayCallback, tensorboard, checkpoint, lr_schedule
from .datalib import SegmentationDataGenerator
from tensorflow.keras.optimizers import Adam

img_width = 352
img_height = 352
log_dir = "logs"
batch_size = 4

filenames = utils.get_segmentation_files()
train_files, val_files = utils.split(filenames, 0.75)

unet = sm.Unet(backbone_name='mobilenetv2',
                   input_shape=(img_width, img_height, 3),
                   encoder_weights='imagenet',
                   decoder_filters=(64,),
                   classes=1)

mask_shape = unet.output.shape[1:3]
mask_width, mask_height = mask_shape[0], mask_shape[1]
print("Output image shape", mask_shape)

train_data = SegmentationDataGenerator(train_files,
                                        img_width=img_width,
                                        img_height=img_height,
                                        mask_width=mask_width,
                                        mask_height=mask_height,
                                        batch_size=batch_size,
                                        shuffle=True)

val_data = SegmentationDataGenerator(val_files,
                                     img_width=img_width,
                                     img_height=img_height,
                                     mask_width=mask_width,
                                     mask_height=mask_height,
                                     batch_size=batch_size,
                                     shuffle=True)


unet.compile(optimizer=Adam(learning_rate=0.001),
              loss=sm.losses.binary_focal_jaccard_loss,
              metrics=[sm.metrics.iou_score])

sample = val_data.sample()

callbacks = [checkpoint("segmentation.hdf5"),
             lr_schedule(),
             tensorboard(),
             DisplayCallback(sample, d_size=600, d_time=1, frequency=1)]

print(len(train_data), len(val_data))

unet.fit(x=train_data, validation_data=val_data, epochs=100, callbacks=callbacks)