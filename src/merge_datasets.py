from .datasets import datasets, get_dataset
import cv2
import json
import os
from tqdm import tqdm

def merge(dsnames, outdir):
    total = 0
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
            image_path = f"{dsname}/{counter}.jpg"
            data_dict[image_path] = all_keypoints
            cv2.imwrite(outdir + image_path, image)

        annotation_path = outdir + f"{dsname}.json"
        json.dump(data_dict, open(annotation_path, 'w'))

    print("total", total)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="Output directory where merged dataset will be written", required=True)
    args = parser.parse_args()

    names = datasets.keys()
    merge(names, args.out)