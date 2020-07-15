from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
from .generators import PoseDataGenerator
from datetime import datetime

class Display:
    def __init__(self, maxsize, time):
        self.maxsize = maxsize
        self.time = time
        self.mode = True
        self.writers = {}

    def getimage(self, image):
        size_ratio = self.maxsize / max(image.shape[0], image.shape[1])
        new_size = int(image.shape[1] * size_ratio), int(image.shape[0] * size_ratio)
        return cv2.resize(image, new_size)

    def show(self, image, name, time=None):
        if not self.mode:
            return

        time = time if time is not None else self.time
        cv2.imshow(name, self.getimage(image))
        cv2.waitKey(time)
        return 0

    def save(self, image, name):
        image = self.getimage(image)
        if name not in self.writers:
            writer = cv2.VideoWriter(name,
                                     cv2.VideoWriter_fourcc('M','J','P','G'),
                                     30, (image.shape[1], image.shape[0]))
            self.writers[name] = writer

        self.writers[name].write(image)

    def off(self):
        self.mode = False

    def on(self):
        self.mode = True

class DisplayCallback(keras.callbacks.Callback):
    """
    Callback adds a opencv display window, and displays results of model inference
    samples : samples of batch to run inference on
    d_size : size of display,
    d_time : opencv sleep time after displaying
    frequency : display will run after every 'frequency' number of epochs
    """

    def __init__(self, val_data, d_size=600, d_time=1, frequency=1, num_samples=4, writer='cv2'):
        super().__init__()
        self.num_samples = num_samples
        self.val_data = val_data
        self.sample_index = 0
        self.samples = self.val_data.sample(self.sample_index)
        self.display = Display(d_size, d_time)
        self.display.on()
        self.callcount = 0
        self.frequency = frequency
        self.tb_writer = self.get_tb_writer()

    def get_tb_writer(self):
        logdir = "logs/image/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
        return file_writer_cm

    def tb_write(self, img, step):
        with self.tb_writer.as_default():
            tf.summary.image("Training data", img, step=step)

    def draw_pose(self, image, keypoints, color=(255, 255, 255), radius=5):
        for keypoint in keypoints:
            x, y, s = keypoint
            cv2.circle(image, (x, y), radius, color, -1)
        return image

    def draw(self, number):
        outimages = []
        images, masks = self.samples
        for index in range(self.num_samples):
            image, mask = images[index], masks[index]
            pred_mask = self.model.predict(np.expand_dims((image), axis=0))[0]

            image = image * 255
            width, height = image.shape[0], image.shape[1]

            mask_sum = np.sum(pred_mask, axis=-1)
            mask_sum = mask_sum / np.max(mask_sum)
            mask_sum = mask_sum * 255
            mask_sum = cv2.cvtColor(mask_sum, cv2.COLOR_GRAY2RGB)
            mask_sum = cv2.resize(mask_sum, image.shape[:2])

            # print("Predicted")
            pred_keypoints = PoseDataGenerator.get_keypoints_from_mask(pred_mask, width, height)
            # print("Original")
            orig_keypoints = PoseDataGenerator.get_keypoints_from_mask(mask, width, height)

            # print("Results..")
            # print("Predicted", pred_keypoints)
            # print("Original", orig_keypoints)

            orig_img = cv2.resize(image, image.shape[:2])

            mask = self.draw_pose(orig_img, orig_keypoints)
            pred_mask = self.draw_pose(orig_img, pred_keypoints, color=(0, 255, 0), radius=4)

            pred_mask = cv2.resize(pred_mask, image.shape[:2])
            mask = cv2.resize(mask, image.shape[:2])

            outimage = np.hstack((pred_mask, mask_sum))
            outimage = cv2.cvtColor(outimage, cv2.COLOR_BGR2RGB)
            outimage = np.asarray(outimage, dtype=np.uint8)
            outimages.append(outimage)

        outimage = np.vstack(outimages)
        outimage = np.expand_dims(outimage, 0)
        #cv2.putText(outimage, str(number), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #self.display.show(outimage, "zorro")
        self.tb_write(outimage, number)

    def on_epoch_end(self, epoch, logs=None):
        self.draw(epoch)

def lr_schedule():
    def lrs(epoch):
        lrs = [0.001, 0.0001, 0.00005, 0.001, 0.0001, 0.00005, 0.00001]
        if epoch < 150:
            return lrs[(epoch // 5) % 7]
        lrs = [0.0001, 0.00005, 0.001, 0.0001, 0.00005, 0.00001]
        return lrs[(epoch // 5) % 6]

    return keras.callbacks.LearningRateScheduler(lrs, verbose=True)

def checkpoint(filepath):
    return keras.callbacks.ModelCheckpoint(filepath=filepath,
                                        monitor='val_loss',
                                        save_best_only=True,
                                        verbose=1)


def tensorboard():
    logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    return tensorboard_callback
