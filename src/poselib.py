import numpy as np
import cv2

class Pose:
    head = 0

    l_ankle = 1
    l_knee = 2
    l_shoulder = 3
    l_wrist = 4
    l_hip = 5
    l_elbow = 6

    r_ankle = 7
    r_knee = 8
    r_shoulder = 9
    r_wrist = 10
    r_hip = 11
    r_elbow = 12

    num_keypoints = 13
