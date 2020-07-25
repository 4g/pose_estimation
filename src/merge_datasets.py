from .datasets import datasets, get_dataset
import cv2
import json
import os
from tqdm import tqdm
import numpy as np

def merge(dsnames, outdir, size):
    total = 0
    size = int(size)
    for dsname in dsnames:
        ds = get_dataset(dsname)
        print(dsname)
        os.mkdir(outdir + dsname)
        data_dict = {}
        counter = 0
        for image_id in tqdm(ds.iter_dataset()):
            counter += 1
            image = ds.get_image(image_id)
            all_keypoints = list(ds.get_keypoints(image_id))
            image, all_keypoints = reduce_size(image, all_keypoints, size)

            image_path = f"{dsname}/{counter}.jpg"
            data_dict[image_path] = all_keypoints
            cv2.imwrite(outdir + image_path, image)

        annotation_path = outdir + f"{dsname}.json"
        json.dump(data_dict, open(annotation_path, 'w'))

    print("total", total)

def reduce_size(image, instances, size):
    ratio = 256 / max(image.shape[0], image.shape[1])
    new_width = int(ratio * image.shape[0])
    new_height = int(ratio * image.shape[1])
    image = cv2.resize(image, (new_height, new_width))

    all_keypoints = []
    num_instances = len(instances)
    num_keypoints = len(instances[0])

    for keypoints in instances:
        all_keypoints.extend(keypoints)

    # rescale keypoints
    all_rescaled_keypoints = []
    for keypoint in all_keypoints:
        x, y, s = keypoint
        x = int(ratio * x)
        y = int(ratio * y)
        keypoint = [x, y, s]
        all_rescaled_keypoints.append(keypoint)

    instances = np.reshape(all_rescaled_keypoints, (num_instances, num_keypoints, 3))
    instances = instances.tolist()
    return image, instances


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Output directory where merged dataset will be written", required=True)
    parser.add_argument("--size", help="reduced image size (largest side in the image will be reduced to this)", required=True)

    args = parser.parse_args()

    names = ["all_merged"]
    merge(names, args.out, args.size)