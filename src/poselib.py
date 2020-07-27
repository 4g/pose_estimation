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

    symmetric_pairs = [
        (l_ankle, r_ankle),
        (l_knee, r_knee),
        (l_shoulder, r_shoulder),
        (l_wrist, r_wrist),
        (l_hip, r_hip),
        (l_elbow, r_elbow),
    ]

    left_points = set()
    right_points = set()

    for i, j in symmetric_pairs:
        left_points.add(i)
        right_points.add(j)

    neutral_points = (set(range(num_keypoints)) - left_points) - right_points

    @staticmethod
    def orientation(index):
        """
        returns -1 if left
        returns 0 if neutral
        returns 1 if right
        """
        if index in Pose.left_points:
            return -1
        if index in Pose.right_points:
            return 1
        if index in Pose.neutral_points:
            return 0