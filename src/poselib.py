import numpy as np
import cv2

class Pose:
    REGISTER = {
        0: "R_Ankle",
        1: "R_Knee",
        2: "R_Hip",
        3: "L_Hip",
        4: "L_Knee",
        5: "L_Ankle",
        6: "B_Pelvis",
        7: "B_Spine",
        8: "B_Neck",
        9: "B_Head",
        10: "R_Wrist",
        11: "R_Elbow",
        12: "R_Shoulder",
        13: "L_Shoulder",
        14: "L_Elbow",
        15: "L_Wrist",
    }

    def __init__(self):
        num_keypoints = len(Pose.REGISTER)
        self.keypoints = np.zeros((num_keypoints, 3))

    def set_keypoint(self, index, x, y, s):
        self.keypoints[index] = [x,y,s]

    def get_keypoints(self):
        return self.keypoints

    def render_pose(self, image, threshold):
        for keypoint in self.keypoints:
            x, y, s = keypoint
            if s >= threshold:
                cv2.circle(image, (x,y), 3, (0,255,0), -1)

        return image