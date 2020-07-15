import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
import segmentation_models as sm

from . import datasets
from .callbacks import DisplayCallback, tensorboard, checkpoint, lr_schedule
from .generators import PoseDataGenerator
from . import modellib

def train(train_iter, val_iter, img_width, img_height, batch_size, model, epochs):
    mask_shape = model.output.shape[1:3]
    mask_width, mask_height = mask_shape[0], mask_shape[1]
    print("Output image shape", mask_shape)

    train_data = PoseDataGenerator(train_iter,
                                   img_width=img_width,
                                   img_height=img_height,
                                   mask_width=mask_width,
                                   mask_height=mask_height,
                                   batch_size=batch_size,
                                   shuffle=True)

    val_data = PoseDataGenerator(val_iter,
                                 img_width=img_width,
                                 img_height=img_height,
                                 mask_width=mask_width,
                                 mask_height=mask_height,
                                 batch_size=batch_size,
                                 shuffle=False)

    print(len(train_data), len(val_data))

    callbacks = [checkpoint("pose_estimation_tanh.hdf5"),
                 lr_schedule(),
                 tensorboard(),
                 DisplayCallback(train_data,
                                d_size=800,
                                d_time=50,
                                frequency=250)]


    model.fit(x=train_data,
              validation_data=val_data,
              epochs=epochs,
              callbacks=callbacks,
              use_multiprocessing=False,
              workers=1,
              max_queue_size=10)

    model.save("pose_saved_model")


def main():
    train_iter = datasets.get_dataset("lip")
    val_iter = datasets.get_dataset("lip")
    # train_iter.set_size(200)
    val_iter.set_size(1000)

    num_keypoints = train_iter.get_num_keypoints()

    img_width = 480
    img_height = 480
    batch_size = 4

    model = modellib.create_pose_model(img_width, img_height, num_keypoints)

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=mse,
                  metrics=[sm.metrics.iou_score])

    # model = keras.models.load_model("pose_estimation_tanh.hdf5", custom_objects={"iou_score": sm.metrics.iou_score})
    model.summary()


    train(train_iter=train_iter,
          val_iter=val_iter,
          img_width=img_width,
          img_height=img_height,
          batch_size=batch_size,
          model=model,
          epochs=300)

    train_iter = datasets.get_dataset("mpii")
    val_iter = datasets.get_dataset("mpii")
    val_iter.set_size(1000)

    train(train_iter=train_iter,
          val_iter=val_iter,
          img_width=img_width,
          img_height=img_height,
          batch_size=batch_size,
          model=model,
          epochs=300)

if __name__ == '__main__':
    main()
