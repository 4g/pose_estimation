from tensorflow import keras
import numpy as np
import cv2


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

        print("Disc size for gaussian:", self.disc_size)


    def get_gaussian(self, size):
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        d = np.sqrt(x * x + y * y)
        sigma, mu = .5, 0.0
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        g = (g - np.min(g))/(np.max(g) - np.min(g))
        return g

    def make_keypoint_mask(self, all_keypoints, width, height):
        keypoint_mask = np.zeros((self.mask_width, self.mask_height, self.num_keypoints*3), dtype=np.float32)

        offset_x_max = self.img_width / self.mask_width
        offset_y_max = self.img_height / self.mask_height

        for keypoints in all_keypoints:
            for index, keypoint in enumerate(keypoints):
                x, y, v = keypoint
                if v == 'nan':
                    continue

                x, y = int(x), int(y)
                scaled_x = int((x/width)*self.mask_width)
                scaled_y = int((y/height)*self.mask_height)

                left_corner_y = max(scaled_y - self.disc_size, 0)
                left_corner_x = max(scaled_x - self.disc_size, 0)

                offset_x = (x - scaled_x * width/self.mask_width) / offset_x_max
                offset_y = (y - scaled_y * height/self.mask_height) / offset_y_max

                sub_mask_shape = (keypoint_mask[left_corner_y: left_corner_y + 2 * self.disc_size + 1,
                              left_corner_x: left_corner_x + 2 * self.disc_size + 1,
                              index]).shape

                keypoint_mask[left_corner_y: left_corner_y + 2 * self.disc_size + 1,
                              left_corner_x: left_corner_x + 2 * self.disc_size + 1,
                              index] = self.gaussian[:sub_mask_shape[0], :sub_mask_shape[1]]

                keypoint_mask[left_corner_y: left_corner_y + 2 * self.disc_size + 1,
                              left_corner_x: left_corner_x + 2 * self.disc_size + 1,
                              index + self.num_keypoints] = offset_x

                keypoint_mask[left_corner_y: left_corner_y + 2 * self.disc_size + 1,
                                left_corner_x: left_corner_x + 2 * self.disc_size + 1,
                                index + 2 * self.num_keypoints] = offset_y



        return keypoint_mask

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
                                          cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0))

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

    def get_img_mask(self, sample):
        img = self.poseiterator.get_image(sample)
        keypoints = list(self.poseiterator.get_keypoints(sample))

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
        maskb = np.zeros((self.batch_size, self.mask_width, self.mask_height, self.num_keypoints*3), dtype=np.float32)

        # Read filenames and fill the batch
        for index, sample in enumerate(batch):
            img, kp = self.get_img_mask(sample)
            imgb[index] = img
            maskb[index] = kp

        return imgb, maskb

    def preprocess_image(self, img, width, height):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height))
        img = img / 255.
        return img

    def sample(self, i=0):
        return self[i]

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)


    @staticmethod
    def get_keypoints_from_mask(mask, width, height):
        num_keypoints = mask.shape[-1]//3
        pred_keypoints = []

        offset_x_max = width / mask.shape[0]
        offset_y_max = height / mask.shape[1]

        for index in range(num_keypoints):
            frame = mask[:, :, index]
            keypoint = np.unravel_index(np.argmax(frame, axis=None), frame.shape)
            y, x = keypoint

            x_scaled = int(x*(width / mask.shape[0]))
            y_scaled = int(y*(width / mask.shape[1]))

            score = frame[keypoint]
            frame = mask[:, :, index + num_keypoints]
            offset_x = frame[keypoint]

            frame = mask[:, :, index + 2 * num_keypoints]
            offset_y = frame[keypoint]

            # print(offset_x, offset_y)

            x = int(x_scaled + offset_x * offset_x_max)
            y = int(y_scaled + offset_y * offset_y_max)

            pred_keypoints.append([x, y, score])

        return pred_keypoints


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    from . import datasets

    train_iter = datasets.get_dataset("lip")
    train_iter.set_size(1000)

    num_keypoints = train_iter.get_num_keypoints()
    print(num_keypoints)

    mask_shape = (15, 15)
    mask_width, mask_height = mask_shape[0], mask_shape[1]
    print("Output image shape", mask_shape)

    img_width = 480
    img_height = 480
    batch_size = 1

    train_data = PoseDataGenerator(train_iter,
                                    img_width=img_width,
                                    img_height=img_height,
                                    mask_width=mask_width,
                                    mask_height=mask_height,
                                    batch_size=batch_size,
                                    shuffle=True)

    for imgb, maskb in train_data:
        img = imgb[0]
        mask = maskb[0]
        keypoints = train_data.get_keypoints_from_mask(mask, img_width, img_height)
        print(keypoints)
        img = train_data.draw_pose(img, keypoints)
        cv2.imshow("img", img)
        cv2.waitKey(-1)
