from .small_model import camera
from .generators import PoseDataGenerator
from tensorflow import keras
import segmentation_models as sm
import cv2
import numpy as np

def test_pose(video, model_path):
    cam = camera.Camera(video)
    model = keras.models.load_model(model_path)

    cam.start()
    while True:
        frame, count = cam.get()
        frame = PoseDataGenerator.crop(frame,  224, 224)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = frame / 127.5 - 1
        image = np.expand_dims(image, axis=0)
        output = model.predict(image)[0]

        output[:,:,14:] = 0
        compressed_output = np.sum(output[:,:,:19], axis=-1)
        # print(np.asarray(compressed_output*32, np.int32))
        compressed_output = compressed_output / np.max(compressed_output)

        compressed_output = np.asarray(compressed_output * 255, np.uint8)

        cv2.imshow("mask", compressed_output)
        keypoints = PoseDataGenerator.get_keypoints_from_mask(output, 224, 224)
        for keypoint in keypoints:
            x, y, s = keypoint
            x = int(x)
            y = int(y)
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.imshow("image", frame)

        cv2.waitKey(10)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="pass a sample image", required=True)
    parser.add_argument("--model", help="pass a model file", required=True)

    args = parser.parse_args()
    model = test_pose(args.video, args.model)
