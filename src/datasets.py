import pycocotools.mask as masklib
import json
from pathlib import Path
import cv2
import numpy as np
from .settings import POSE_DATASET_DIR
import os
from scipy.io import loadmat


class PoseDataSource:
    KEYPOINTS = 'keypoints'
    FILE_PATH = 'file_path'
    ANNOTATIONS = 'annotations'

    def __init__(self, annotation_path, images_dir):
        self.images_dir = Path(images_dir)
        self.annotation_path = annotation_path

        self.num_keypoints = None
        self.data_map = {}
        self.create_data_map()
        self.images_ids = list(self.data_map.keys())
        self.itersize = len(self.images_ids)

        # Check and autoset number of keypoints if not already set
        if self.num_keypoints is None:
            self.set_num_keypoints()

    def create_data_map(self):
        """
        1. Process the annotation file
        2. Return a dictionary(d), where keys are
            imageids, i.e. each key corresponds to one image
        3. d[image_id] should be a dictionary, with key=PoseDataSource.ANNOTATIONS
        4. Since there can be multiple annotations per imageid:
            a. key = PoseDataSource.KEYPOINTS, value = list of keypoints of shape (num_keypoints, 3)
            b. each keypoint should be (x,y,v) : where v represents visibility, leave as 0(zero) if not available

        example :
        data_map =
        { image_id1 : {'annotations': [{'keypoints':[(1,1,0), ...], .. }, {'keypoints':[(1,1,0), ...]}, {'keypoints':[(1,1,0), ...]}],
            ..
            ..
        }
        """
        return {}

    def has(self, image_id):
        return image_id in self.data_map

    def remove(self, image_id):
        if self.has(image_id):
            del self.data_map[image_id]

    def get_keypoints(self, image_id):
        sample = self.get_sample(image_id)
        instances = sample[PoseDataSource.ANNOTATIONS]
        for instance in instances:
            keypoints = instance[PoseDataSource.KEYPOINTS]
            yield keypoints

    def get_image(self, image_id):
        return cv2.imread(str(self.get_filepath(image_id)))

    def get_filepath(self, image_id):
        return self.images_dir / self.data_map[image_id][self.FILE_PATH]

    def iter_dataset(self, size=None):
        size = size or self.itersize
        for image_id in self.images_ids[:size]:
            yield image_id

    @staticmethod
    def annotation_format():
        return {PoseDataSource.ANNOTATIONS: [],
                                   PoseDataSource.FILE_PATH: None}

    def add_image_id(self, image_id):
        if self.has(image_id):
            return
        self.data_map[image_id] = PoseDataSource.annotation_format()

    def set_filename(self, image_id, filename):
        self.data_map[image_id][PoseDataSource.FILE_PATH] = filename

    def add_annotation(self, image_id, keypoints):
        annotation = {PoseDataSource.KEYPOINTS:keypoints}
        self.data_map[image_id][PoseDataSource.ANNOTATIONS].append(annotation)

    def get_sample(self, image_id):
        return self.data_map[image_id]

    def render_image(self, image_id):
        all_keypoints = self.get_keypoints(image_id)
        image = self.get_image(image_id)
        for keypoints in all_keypoints:
            for keypoint in keypoints:
                x, y, v = keypoint
                if v == 'nan':
                    continue
                x = int(x)
                y = int(y)
                cv2.circle(image, center=(x, y), radius=3, color=(255, 255, 255), thickness=-1)
        return image

    def get_num_keypoints(self):
        return self.num_keypoints

    def set_num_keypoints(self, value=None):
        if value is None:
            kp = next(self.get_keypoints(self.images_ids[0]))
            value = len(kp)
        self.num_keypoints = value

    def set_size(self, size):
        self.itersize = size

class Coco(PoseDataSource):
    IMGS = 'images'
    IMAGE_ID = 'image_id'
    ANNO = 'annotations'

    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        self.raw_data = json.load(open(self.annotation_path))

        for elem in self.raw_data[Coco.ANNO]:
            image_id = elem[self.IMAGE_ID]
            keypoints = elem[Coco.KEYPOINTS]
            keypoints = np.reshape(keypoints, (17,3))
            self.add_image_id(image_id)
            self.add_annotation(image_id, keypoints)

        for elem in self.raw_data[Coco.IMGS]:
            image_id = elem['id']
            filename = elem['file_name']
            if self.has(image_id):
                self.set_filename(image_id, filename)

    def get_segmentation_mask(self, sample):
        height = sample['height']
        width = sample['width']
        seg_mask = None

        for instance in sample[self.ANNO]:
            rle = masklib.frPyObjects(instance['segmentation'], height, width)
            mask = masklib.decode(rle)
            if len(mask.shape) == 3:
                mask = np.sum(mask, axis=-1, dtype=np.uint8)

            if seg_mask is not None:
                seg_mask = mask + seg_mask
            else:
                seg_mask = mask

        return seg_mask.astype(np.uint8) * 255

class MPII(PoseDataSource):
    IMGS = 'images'
    IMAGE_ID = 'image'
    KEYPOINTS = "joints"

    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        data_map = {}
        self.raw_data = json.load(open(self.annotation_path))

        for elem in self.raw_data:
            image_id = elem[self.IMAGE_ID]
            keypoints = elem[MPII.KEYPOINTS]

            # add the visibility index
            for kp in keypoints:
                kp.append(0)

            keypoints = np.reshape(keypoints, (16, 3))

            self.add_image_id(image_id)
            self.add_annotation(image_id, keypoints)
            self.set_filename(image_id, image_id)

class LIP(PoseDataSource):
    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        self.raw_data = [l.strip().split(",") for l in open(self.annotation_path)]

        for elem in self.raw_data:
            image_id = elem[0]
            keypoints = elem[1:]
            keypoints = np.asarray(keypoints, dtype=np.float32)
            keypoints = [self.to_int(x) for x in keypoints]
            keypoints = np.reshape(keypoints, (16, 3))
            self.add_image_id(image_id)
            self.add_annotation(image_id, keypoints)
            self.set_filename(image_id, image_id)

    def to_int(self, x):
        if np.isnan(x):
            return -1
        return float(x)

class LSP(PoseDataSource):
    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        from scipy.io import loadmat
        annotations = loadmat(self.annotation_path)
        self.raw_data = annotations['joints']
        num_samples = self.raw_data.shape[-1]
        for index in range(num_samples):
            joints = self.raw_data[:,:,index]
            image_id = "im{index}.jpg".format(index=str(index + 1).zfill(5))
            self.add_image_id(image_id)
            self.add_annotation(image_id, joints)
            self.set_filename(image_id, image_id)

class CROWD(PoseDataSource):
    IMGS = 'images'
    IMAGE_ID = 'image_id'
    ANNO = 'annotations'

    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        self.raw_data = json.load(open(self.annotation_path))

        for elem in self.raw_data[Coco.ANNO]:
            image_id = elem[self.IMAGE_ID]
            keypoints = elem[Coco.KEYPOINTS]
            keypoints = np.reshape(keypoints, (14,3))
            self.add_image_id(image_id)
            self.add_annotation(image_id, keypoints)

        for elem in self.raw_data[Coco.IMGS]:
            image_id = elem['id']
            filename = elem['file_name']
            if self.has(image_id):
                self.set_filename(image_id, filename)

class SURREAL(PoseDataSource):
    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        mats = set()
        mp4s = set()
        for root, directories, files in os.walk(self.annotation_path):
            for file in files:
                file_ext = file.split(".")[1]
                if file_ext == 'mp4':
                    mp4s.add((root, file))

        for root, file in mp4s:
            annot = file.replace(".mp4", "_info.mat")
            path = Path(root)
            annotation = loadmat(str(path/annot))
            joints = annotation['joints2D']

class Posetrack(PoseDataSource):
    IMGS = 'images'
    IMAGE_ID = 'image_id'
    ANNO = 'annotations'

    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        import glob
        x = 0
        for f in glob.glob(self.annotation_path + "*.json"):
            annotation = json.load(open(f))
            x += len(annotation[Coco.ANNO])
            for elem in annotation[Coco.ANNO]:
                image_id = elem[self.IMAGE_ID]
                keypoints = elem[Coco.KEYPOINTS]
                keypoints = np.reshape(keypoints, (17, 3))
                self.add_image_id(image_id)
                self.add_annotation(image_id, keypoints)

            for elem in annotation[Coco.IMGS]:
                # print(elem)
                image_id = elem['id']
                filename = elem['file_name']
                if self.has(image_id):
                    self.set_filename(image_id, filename)
                    path = self.get_filepath(image_id)
                    if not os.path.exists(path):
                        self.remove(image_id)

        print("Num images", len(self.data_map), x)

class MergedDataSource(PoseDataSource):
    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)


def get_dataset(name):
   datasets = {
    "lsp": (LSP,
           "leeds_sports/lspet_dataset/joints.mat",
           "leeds_sports/lspet_dataset/images/",
            ),

    "coco": (Coco,
            'coco/annotations/person_keypoints_train2017.json',
            "coco/images/train/",
             ),

    "mpii": (MPII,
           "mpii_pose_photos_dataset/trainval.json",
           "mpii_pose_photos_dataset/images",
           ),

    "lip": (LIP,
           "single_person_coco_hi/TrainVal_pose_annotations/lip_train_set.csv",
           "single_person_coco_hi/TrainVal_images/train_images/",
            ),

    "lip_val": (LIP,
           "single_person_coco_hi/TrainVal_pose_annotations/lip_val_set.csv",
           "single_person_coco_hi/TrainVal_images/val_images/",
           ),

    "crowd": (CROWD,
               "crowd_pose/annotations/json/crowdpose_train.json",
               "crowd_pose/images",
               ),

    "surreal": (SURREAL,
                "surreal/dataset/SURREAL/data/cmu/train/run1/",
                "surreal/dataset/SURREAL/data/cmu/train/run1/"
                ),
    "posetrack" : (Posetrack,
                "posetrack/annotations/train/",
                "posetrack/"
    )
   }

   ds_tuple = datasets[name]
   ds_method = ds_tuple[0]
   ds_annot = POSE_DATASET_DIR + ds_tuple[1]
   ds_images = POSE_DATASET_DIR + ds_tuple[2]
   dataset = ds_method(ds_annot, ds_images)
   return dataset