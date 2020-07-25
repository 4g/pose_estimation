from tensorflow import keras
import numpy as np
import cv2
from tqdm import tqdm
import random
import albumentations as A

class SegmentationDataGenerator(keras.utils.Sequence):
    """
    Initialize with a list  of filename tuples [(img_path0, mask_path0), (img_path1, mask_path1) ...]
    It is a keras sequence object, and can be directly iterated over by keras models, fit methods.
    1] sample() : returns first batch of images
    2] datagen = SegmentationDataGenerator(flist, 32, 320, 320, True)
        datagen[n] : returns nth batch of images
        shuffle : True, shuffle after every epoch
    """

    def __init__(self, filenames, batch_size, img_width, img_height, mask_width, mask_height, shuffle=True):
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.mask_width = mask_width
        self.mask_height = mask_height
        self.file_names = filenames
        self.shuffle = shuffle


    def __len__(self):
        return int(np.ceil(len(self.file_names) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.file_names[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Create an empty batch of images and mask each
        imgb = np.zeros((self.batch_size, self.img_width, self.img_height, 3), dtype=np.float32)
        maskb = np.zeros((self.batch_size, self.mask_width, self.mask_height), dtype=np.float32)

        # Read filenames and fill the batch
        for index, items in enumerate(batch):
            imgpath, maskpath = items
            img = self.get_image(imgpath, self.img_width, self.img_height)
            mask = self.get_image(maskpath, self.mask_width, self.mask_height)[:,:,0]

            # Mask has to be binary
            # convert non zero elements to 1
            mask[mask != 0] = 1.0
            imgb[index] = img
            maskb[index] = mask
        return imgb, maskb

    def get_image(self, path, width, height):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))
        img = img / 255.
        return img

    def sample(self):
        return self[0]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_names)

class PoseDataGenerator(keras.utils.Sequence):
    """
    Initialize with a list of dictionaries, each containing info about image  [{'image_path':path, 'keypoints':keypoints list}]
    This is an indexed object and elements can be accessed by calling self[i]
    1] sample() : returns first batch of images
    2] datagen = PoseDataGenerator(flist, 32, 320, 320, True)
        datagen[n] : returns nth batch of images
        shuffle : True, shuffle after every epoch
    """

    def __init__(self, poseiterator, batch_size, img_width, img_height, mask_width, mask_height, shuffle=True):
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.mask_width = mask_width
        self.mask_height = mask_height
        self.poseiterator = poseiterator
        self.num_keypoints = self.poseiterator.get_num_keypoints()
        self.shuffle = shuffle
        self.data = [sample for sample in self.poseiterator.iter_dataset()]
        self.disc_size = max(int(max(np.sqrt(self.mask_width)/2, 1)), 3)
        self.gaussian = self.get_gaussian(self.disc_size*2 + 1)

        random.seed(7)
        transform = A.Compose([
            A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_REPLICATE),
            A.RandomRotate90(p=0.3),
            # A.RandomSnow(p=0.1),
            # A.MotionBlur(p=0.1),
            A.OneOf([
                A.HueSaturationValue(p=0.3),
                A.RGBShift(p=0.1)
            ], p=0),
            A.RandomBrightnessContrast(p=0.25),
        ],
            keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
        )
        self.transform = transform

        print("Disc size for gaussian:", self.disc_size)


    def get_gaussian(self, size):
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        d = np.sqrt(x * x + y * y)
        sigma, mu = .5, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        g = (g - np.min(g))/(np.max(g) - np.min(g))
        return g

    def make_keypoint_mask(self, all_keypoints, width, height):
        keypoint_mask = np.zeros((self.mask_width, self.mask_height, self.num_keypoints), dtype=np.float32)

        for keypoints in all_keypoints:
            for index, keypoint in enumerate(keypoints):
                x, y, v = keypoint

                if v == -1 or x > self.img_height or y > self.img_width or x <= self.disc_size or y <= self.disc_size:
                    continue

                x, y = int(x), int(y)
                x = min(self.img_height, x)
                y = min(self.img_width, y)

                scaled_x = int((x/width)*self.mask_width)
                scaled_y = int((y/height)*self.mask_height)

                left_space_x = min(scaled_x, self.disc_size)
                left_space_y = min(scaled_y, self.disc_size)

                right_space_x = min(self.mask_width - scaled_x, self.disc_size + 1)
                right_space_y = min(self.mask_height - scaled_y, self.disc_size + 1)

                keypoint_mask[scaled_y - left_space_y: scaled_y + right_space_y,
                              scaled_x - left_space_x: scaled_x + right_space_x,
                              index] = self.gaussian[self.disc_size - left_space_y:self.disc_size + right_space_y,
                                                     self.disc_size - left_space_x:self.disc_size + right_space_x]

        return keypoint_mask

    def get_img_mask(self, sample):
        img = self.poseiterator.get_image(sample)
        instances = list(self.poseiterator.get_keypoints(sample))
        num_instances = len(instances)

        all_keypoints = []
        for keypoints in instances:
            all_keypoints.extend(keypoints)

        bad_indices = []
        good_keypoints = []
        for index, keypoint in enumerate(all_keypoints):
            good_keypoint = keypoint
            if keypoint[0] <= 0 or keypoint[1] <= 0:
                bad_indices.append(index)
                good_keypoint = [0, 0, -1]

            good_keypoints.append(good_keypoint)


        transformed = self.transform(image=img, keypoints=good_keypoints)
        img = transformed['image']
        all_keypoints = transformed['keypoints']

        for index in bad_indices:
            all_keypoints[index] = [-1, -1, -1]

        keypoints = np.reshape(all_keypoints, (num_instances, self.num_keypoints, 3))

        img, keypoints = self.pad(img, keypoints, self.img_width, self.img_height)
        keypoint_mask = self.make_keypoint_mask(keypoints, self.img_width, self.img_height)

        img = self.preprocess_image(img, self.img_width, self.img_height)

        return img, keypoint_mask

    def draw_pose(self, image, keypoints, color=(0, 255, 0), radius=4):
        for keypoint in keypoints:
            x, y, s = keypoint
            x = int(x)
            y = int(y)
            cv2.circle(image, (x, y), radius, color, -1)
        return image

    def __len__(self):
        return int(np.ceil(len(self.data) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.data[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Create an empty batch of images and mask each
        imgb = np.zeros((self.batch_size, self.img_width, self.img_height, 3), dtype=np.float32)
        maskb = np.zeros((self.batch_size, self.mask_width, self.mask_height, self.num_keypoints), dtype=np.float32)

        # Read filenames and fill the batch
        for index, sample in enumerate(batch):
            img, kp = self.get_img_mask(sample)
            imgb[index] = img
            maskb[index] = kp

        return imgb, maskb

    def preprocess_image(self, img, width, height):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))
        img = img / 127.5
        img -= 1.
        return img

    def sample(self, i=0):
        return self[i]

    def shuffle_dataset(self):
        np.random.shuffle(self.data)

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_dataset()

    @staticmethod
    def get_keypoints_from_mask(mask, width, height):
        num_keypoints = mask.shape[-1]
        pred_keypoints = []

        for index in range(num_keypoints):
            frame = mask[:, :, index]
            keypoint = np.unravel_index(np.argmax(frame, axis=None), frame.shape)
            y, x = keypoint

            x_scaled = int(x*(width / mask.shape[0]))
            y_scaled = int(y*(width / mask.shape[1]))

            score = frame[keypoint]

            pred_x = int(x_scaled)
            pred_y = int(y_scaled)

            pred_keypoints.append([pred_x, pred_y, score])

        return pred_keypoints

    @staticmethod
    def crop(image, width, height):
        """
        Crops an image to desired width / height ratio
        :param image: image to crop
        :param width: desired width
        :param height: desired height
        :return: returns an image cropped to width/height ratio
        """
        desired_ratio = width / height
        image_width = image.shape[1]
        image_height = image.shape[0]
        image_ratio = image_width / image_height
        new_width, new_height = image_width, image_height

        # if original image is wider than desired image, crop across width
        if image_ratio > desired_ratio:
            new_width = int(image_height * desired_ratio)

        # crop across height otherwise
        elif image_ratio < desired_ratio:
            new_height = int(image_width / desired_ratio)

        image = image[image_height // 2 - new_height // 2: image_height // 2 + new_height // 2,
                image_width // 2 - new_width // 2: image_width // 2 + new_width // 2]

        image = cv2.resize(image, (width, height))

        return image

    @staticmethod
    def pad(image, all_keypoints, width, height):
        image_width = image.shape[1]
        image_height = image.shape[0]

        resize_ratio = min(width / image_width, height / image_height)
        new_width, new_height = int(resize_ratio * image_width), int(resize_ratio * image_height)
        new_img = cv2.resize(image, (new_width, new_height))
        pad_width = (width - new_width) // 2
        pad_height = (height - new_height) // 2
        padded_image = cv2.copyMakeBorder(new_img,
                                          pad_height,
                                          pad_height,
                                          pad_width,
                                          pad_width,
                                          cv2.BORDER_REPLICATE)

        padded_image = cv2.resize(padded_image, (width, height))
        for index in range(len(all_keypoints)):
            keypoints = all_keypoints[index]
            padded_keypoints = []
            for x,y,s in keypoints:
                new_x = (float(x) * resize_ratio + pad_width)
                new_y = (float(y) * resize_ratio + pad_height)
                padded_keypoints.append([new_x, new_y, s])

            all_keypoints[index] = padded_keypoints

        return padded_image, all_keypoints



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=False, default="lip")
    args = parser.parse_args()
    ds = args.dataset

    from . import datasets

    train_iter = datasets.get_dataset(args.dataset)

    # train_iter.set_size(10)

    num_keypoints = train_iter.get_num_keypoints()
    print(num_keypoints)

    mask_shape = (64, 64)
    mask_width, mask_height = mask_shape[0], mask_shape[1]
    print("Output image shape", mask_shape)

    img_width = 256
    img_height = 256
    batch_size = 1

    train_data = PoseDataGenerator(train_iter,
                                    img_width=img_width,
                                    img_height=img_height,
                                    mask_width=mask_width,
                                    mask_height=mask_height,
                                    batch_size=batch_size,
                                    shuffle=False)
    train_data.shuffle_dataset()

    for imgb, maskb in tqdm(train_data):
        img = imgb[0]
        mask = maskb[0]

        keypoints = train_data.get_keypoints_from_mask(mask, img_width, img_height)
        img = (img + 1)*127
        img = img.astype(np.uint8)
        img = train_data.draw_pose(img, keypoints)
        cv2.imshow("img", img)
        cv2.waitKey(-1)
