import pycocotools.mask as masklib
import json
from pathlib import Path
import cv2
import numpy as np
from .settings import POSE_DATASET_DIR
import os
from scipy.io import loadmat
from .poselib import Pose as P

class PoseDataSource:
    KEYPOINTS = 'keypoints'
    FILE_PATH = 'file_path'
    ANNOTATIONS = 'annotations'
    order = [P.head, P.l_shoulder, P.r_shoulder, P.l_elbow, P.r_elbow, P.l_wrist, P.r_wrist, P.l_hip, P.r_hip, P.l_knee, P.r_knee, P.l_ankle, P.r_ankle]
    def __init__(self, annotation_path, images_dir):
        self.images_dir = Path(images_dir)
        self.annotation_path = annotation_path

        self.num_keypoints = None
        self.data_map = {}
        self.create_data_map()
        self.images_ids = sorted(list(self.data_map.keys()))
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
        keypoints = []
        for instance in instances:
            instance_keypoints = instance[PoseDataSource.KEYPOINTS]
            keypoints.append(instance_keypoints)
        return keypoints

    def get_image(self, image_id):
        return cv2.imread(str(self.get_imgpath(image_id)))

    def get_imgpath(self, image_id):
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

    def clean_keypoint(self, keypoint):
        x, y, v = keypoint
        x = float(x)
        y = float(y)
        v = float(v)
        return [x, y, v]

    def add_annotation(self, image_id, keypoints):
        clean_keypoints = [self.clean_keypoint(kp) for kp in keypoints]
        keypoints = list(range(P.num_keypoints))

        for index, keypoint in enumerate(clean_keypoints):
            correct_index = self.order[index]
            if correct_index != -1:
                keypoints[correct_index] = keypoint

        annotation = {PoseDataSource.KEYPOINTS: keypoints}
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
            kp = self.get_keypoints(self.images_ids[0])[0]
            value = len(kp)
        self.num_keypoints = value

    def set_size(self, size):
        self.itersize = size

class Coco(PoseDataSource):
    IMGS = 'images'
    IMAGE_ID = 'image_id'
    ANNO = 'annotations'
    # ['Nose', 'Leye', 'Reye',
    #  'Lear', 'Rear',
    #  'Lsho', 'Rsho',
    #  'Lelb', 'Relb',
    #  'Lwri', 'Rwri',
    #  'Lhip', 'Rhip',
    #  'Lkne', 'Rkne',
    #  'Lank', 'Rank']

    order = [P.head, -1, -1, -1, -1, P.l_shoulder, P.r_shoulder, P.l_elbow, P.r_elbow, P.l_wrist, P.r_wrist, P.l_hip, P.r_hip, P.l_knee, P.r_knee, P.l_ankle, P.r_ankle ]

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

    """
    1, R_Ankle
    2, R_Knee
    3, R_Hip
    4, L_Hip
    5, L_Knee
    6, L_Ankle
    7, B_Pelvis
    8, B_Spine
    9, B_Neck
    10, B_Head
    11, R_Wrist
    12, R_Elbow
    13, R_Shoulder
    14, L_Shoulder
    15, L_Elbow
    16, L_Wrist
    """
    order = [P.r_ankle, P.r_knee, P.r_hip, P.l_hip, P.l_knee, P.l_ankle, -1, -1, -1, P.head, P.r_wrist, P.r_elbow, P.r_shoulder, P.l_shoulder, P.l_elbow, P.l_wrist]
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
    order = [P.r_ankle, P.r_knee, P.r_hip, P.l_hip, P.l_knee, P.l_ankle, -1, -1, -1, P.head, P.r_wrist, P.r_elbow,
             P.r_shoulder, P.l_shoulder, P.l_elbow, P.l_wrist]

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
    """
    Right ankle
    Right knee
    Right hip
    Left hip
    Left knee
    Left ankle
    Right wrist
    Right elbow
    Right shoulder
    Left shoulder
    Left elbow
    Left wrist
    Neck
    Head top
    """
    order = [P.r_ankle, P.r_knee, P.r_hip, P.l_hip, P.l_knee, P.l_ankle, P.r_wrist, P.r_elbow,
             P.r_shoulder, P.l_shoulder, P.l_elbow, P.l_wrist, -1, P.head]

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

    """
    ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 
    'left_ankle', 'right_ankle', 'head', 'neck'],
    """
    order = [P.l_shoulder, P.r_shoulder, P.l_elbow, P.r_elbow, P.l_wrist, P.r_wrist, P.l_hip, P.r_hip, P.l_knee, P.r_knee, P.l_ankle, P.r_ankle, P.head, -1]
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

    """
     "nose", 
    "head_bottom", 
    "head_top", 
    "left_ear", 
    "right_ear", 
    "left_shoulder", 
    "right_shoulder", 
    "left_elbow", 
    "right_elbow", 
    "left_wrist", 
    "right_wrist", 
    "left_hip", 
    "right_hip", 
    "left_knee", 
    "right_knee", 
    "left_ankle", 
    "right_ankle"
    """
    order = [-1, -1, P.head, -1, -1, P.l_shoulder, P.r_shoulder, P.l_elbow, P.r_elbow, P.l_wrist, P.r_wrist, P.l_hip, P.r_hip, P.l_knee, P.r_knee, P.l_ankle, P.r_ankle]

    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)
    #
    # def create_data_map(self):
    #     import glob
    #     for f in glob.glob(self.annotation_path + "*.json"):
    #         annotation = json.load(open(f))
    #         for elem in annotation[Coco.ANNO]:
    #             image_id = elem[self.IMAGE_ID]
    #             keypoints = elem[Coco.KEYPOINTS]
    #             keypoints = np.reshape(keypoints, (17, 3))
    #             self.add_image_id(image_id)
    #             self.add_annotation(image_id, keypoints)
    #
    #         for elem in annotation[Coco.IMGS]:
    #             # print(elem)
    #             image_id = elem['id']
    #             filename = elem['file_name']
    #             if self.has(image_id):
    #                 self.set_filename(image_id, filename)
    #                 path = self.get_imgpath(image_id)
    #                 if not os.path.exists(path):
    #                     self.remove(image_id)

    def create_data_map(self):
        import glob
        for annotation_path in glob.glob(self.annotation_path + "*.json"):
            annotations = json.load(open(annotation_path))
            for image_path in annotations:
                instances = annotations[image_path]
                image_id = image_path
                self.add_image_id(image_id)
                for instance in instances:
                    self.add_annotation(image_id, instance)
                self.set_filename(image_id, image_path)

class Penn(PoseDataSource):
    """
    1.  head
    2.  left_shoulder  3.  right_shoulder
    4.  left_elbow     5.  right_elbow
    6.  left_wrist     7.  right_wrist
    8.  left_hip       9.  right_hip
    10. left_knee      11. right_knee
    12. left_ankle     13. right_ankle
    """
    order = [P.head, P.l_shoulder, P.r_shoulder, P.l_elbow, P.r_elbow, P.l_wrist, P.r_wrist, P.l_hip, P.r_hip, P.l_knee, P.r_knee, P.l_ankle, P.r_ankle]

    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        from scipy.io import loadmat
        import glob
        for f in glob.glob(self.annotation_path + "*.mat"):
            annotation = loadmat(f)
            video_index = Path(f).name.split(".")[0]
            all_x = annotation['x']
            all_y = annotation['y']
            all_visibility = annotation['visibility']
            num_frames = int(annotation['nframes'])

            for index in range(num_frames):
                x = all_x[index]
                y = all_y[index]
                visibility = all_visibility[index]
                image_name = '{0:06}'.format(index + 1) + ".jpg"
                image_path = video_index + "/" + image_name
                keypoints = list(zip(x, y, visibility))
                keypoints = np.asarray(keypoints, dtype=np.float32)

                image_id = image_path
                self.add_image_id(image_id)
                self.add_annotation(image_id, keypoints)
                self.set_filename(image_id, image_id)

class MergedPosetrack(PoseDataSource):
    order = [-1, -1, P.head, -1, -1, P.l_shoulder, P.r_shoulder, P.l_elbow, P.r_elbow, P.l_wrist, P.r_wrist, P.l_hip,
             P.r_hip, P.l_knee, P.r_knee, P.l_ankle, P.r_ankle]

    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        import glob
        for annotation_path in glob.glob(self.annotation_path + "*.json"):
            annotations = json.load(open(annotation_path))
            for image_path in annotations:
                instances = annotations[image_path]
                image_id = image_path
                self.add_image_id(image_id)
                for instance in instances:
                    self.add_annotation(image_id, instance)
                self.set_filename(image_id, image_path)

    def get_keypoints(self, image_id):
        sample = self.get_sample(image_id)
        instances = sample[PoseDataSource.ANNOTATIONS]
        keypoints = []
        for instance in instances:
            instance_keypoints = instance[PoseDataSource.KEYPOINTS]
            keypoints.append(instance_keypoints)
        return keypoints

class MergedDataSource(PoseDataSource):

    def __init__(self, annotation_path, images_dir):
        super().__init__(annotation_path, images_dir)

    def create_data_map(self):
        import glob
        for annotation_path in glob.glob(self.annotation_path + "*.json"):
            annotations = json.load(open(annotation_path))
            for image_path in annotations:
                instances = annotations[image_path]
                image_id = image_path
                self.add_image_id(image_id)
                for instance in instances:
                    self.add_annotation(image_id, instance)
                self.set_filename(image_id, image_path)

    def get_keypoints(self, image_id):
        sample = self.get_sample(image_id)
        instances = sample[PoseDataSource.ANNOTATIONS]
        keypoints = []
        for instance in instances:
            instance_keypoints = instance[PoseDataSource.KEYPOINTS]
            keypoints.append(instance_keypoints)
        return keypoints

datasets = {
        "lsp": (LSP,
               "leeds_sports/lspet_dataset/joints.mat",
               "leeds_sports/lspet_dataset/images/",
                ),

        "coco": (Coco,
                'coco/annotations/person_keypoints_train2017.json',
                "coco/images/train/",
                 ),
        "coco_val":(Coco,
                'coco/annotations/person_keypoints_val2017.json',
                "coco/images/val/",
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
                   "crowd_pose/images/",
                   ),

        "crowd_val": (CROWD,
                 "crowd_pose/annotations/json/crowdpose_val.json",
                 "crowd_pose/images/",
                 ),

        # "surreal": (SURREAL,
        #             "surreal/dataset/SURREAL/data/cmu/train/run1/",
        #             "surreal/dataset/SURREAL/data/cmu/train/run1/"
        #             ),

        # "posetrack": (Posetrack,
        #             "posetrack/annotations/train/",
        #             "posetrack/"
        #             ),
        #
        # "posetrack_val": (Posetrack,
        #              "posetrack/annotations/val/",
        #              "posetrack/"
        #              ),

        "penn": (Penn,
                  "Penn_Action/labels/",
                  "Penn_Action/frames/"
                  ),

        "posetrack": (MergedPosetrack,
                 "merged/posetrack/",
                 "merged/posetrack/"),

        "all_merged": (MergedDataSource,
                  "merged/all_merged/",
                  "merged/all_merged/"),
}

def get_dataset(name):
   ds_tuple = datasets[name]
   ds_method = ds_tuple[0]
   ds_annot = POSE_DATASET_DIR + ds_tuple[1]
   ds_images = POSE_DATASET_DIR + ds_tuple[2]
   dataset = ds_method(ds_annot, ds_images)
   return dataset


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Output directory where merged dataset will be written", required=True)
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    for image_id in ds.iter_dataset():
        image = ds.render_image(image_id)
        cv2.imshow("image", image)
        cv2.waitKey(-1)