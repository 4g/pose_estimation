import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse

from . import datasets
from .callbacks import DisplayCallback, tensorboard, checkpoint, lr_schedule
from .generators import PoseDataGenerator
from . import modellib

def mixed_precision(prec):
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy(prec)
    mixed_precision.set_policy(policy)

def train(train_iter, val_iter, img_width, img_height, batch_size, model, epochs, model_path):
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

    callbacks = [checkpoint(model_path),
                 lr_schedule(),
                 tensorboard()]

    callbacks.append(DisplayCallback(train_data,
                    d_size=800,
                    d_time=50,
                    frequency=250))


    model.fit(x=train_data,
              validation_data=val_data,
              epochs=epochs,
              callbacks=callbacks,
              use_multiprocessing=False,
              workers=1,
              max_queue_size=10)

    model.save("pose_saved_model")


def main(train_ds, val_ds, prec, model_prefix, sample_size):

    mixed_precision(prec)

    train_iter = datasets.get_dataset(train_ds)
    val_iter = datasets.get_dataset(val_ds)
    train_iter.set_size(sample_size)
    val_iter.set_size(sample_size)

    num_keypoints = train_iter.get_num_keypoints()

    img_width = 480
    img_height = 480
    batch_size = 4

    model = modellib.create_pose_model(img_width, img_height, num_keypoints)

    model.compile(optimizer=Adam(learning_rate=0.01),
                  loss=mse,
                  metrics=['mse'])

    model.summary()

    model_filename = model_prefix + "{epoch:02d}-{val_loss:.5f}.hdf5"

    train(train_iter=train_iter,
          val_iter=val_iter,
          img_width=img_width,
          img_height=img_height,
          batch_size=batch_size,
          model=model,
          epochs=100,
          model_path=model_filename)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prec", help="mixed precision mode (mixed_float16 / float32)", required=False, default="mixed_float16")
    parser.add_argument("--out", help="output file prefix for modelname", required=False, default="runs/model.")
    parser.add_argument("--train",  help="name of datasets to use for training", required=False, default="lip")
    parser.add_argument("--val", help="name of datasets to use for training", required=False, default="lip_val")
    parser.add_argument("--sample", help="sample size, give small sample size to test code", required=False, default=None, type=int)

    args = parser.parse_args()
    main(args.train, args.val, args.prec, args.out, args.sample)