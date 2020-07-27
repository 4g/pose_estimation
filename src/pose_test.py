from .small_model import camera
from .generators import PoseDataGenerator
from tensorflow import keras
import segmentation_models as sm
import cv2
import numpy as np


def extract_mask(output, start=0, end=None):
    compressed_output = np.sum(output[:,:,start:end], axis=-1)
    # print(np.asarray(compressed_output*32, np.int32))
    compressed_output = compressed_output / np.max(compressed_output)
    compressed_output = np.asarray(compressed_output * 255, np.uint8)
    return compressed_output

def test_pose(video, model_path):
    cam = camera.Camera(video, fps=100)
    model = keras.models.load_model(model_path)

    w, h = 256, 256
    first_time = True

    cam.start()
    while True:
        frame, count = cam.get()
        # frame = cv2.transpose(frame)
        # frame, _ = PoseDataGenerator.pad(frame,[], w, h)
        frame = cv2.resize(frame, (w, h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = frame / 127.5 - 1
        image = np.expand_dims(image, axis=0)
        output = model.predict(image)[0]

        compressed_output_1 = extract_mask(output, 1, 7)
        compressed_output_2 = extract_mask(output, 7, 13)
        compressed_output_3 = extract_mask(output, 0, 1)
        ow, oh = compressed_output_1.shape

        compressed_output = np.zeros((ow, oh, 3), dtype=np.uint8)
        compressed_output[:, :, 0] = compressed_output_1
        compressed_output[:, :, 1] = compressed_output_2
        compressed_output[:, :, 2] = compressed_output_3

        # compressed_output[compressed_output < 80] = 0

        compressed_output = cv2.resize(compressed_output, (w, h))

        #        cv2.imshow("mask", compressed_output)
        keypoints = PoseDataGenerator.get_keypoints_from_mask(output, w, h)
        for keypoint in keypoints:
            x, y, s = keypoint
            x = int(x)
            y = int(y)
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)
        output = np.hstack((frame, compressed_output))
        cv2.imshow("predicted", output)
        cv2.waitKey(10)

        if first_time:
            cv2.waitKey(-1)
            first_time = False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="pass a sample image", required=True)
    parser.add_argument("--model", help="pass a model file", required=True)

    args = parser.parse_args()
    model = test_pose(args.video, args.model)
