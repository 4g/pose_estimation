import numpy as np
import cv2

class Pose:
    __slots__ = 'R_Ankle','R_Knee',\
                'R_Hip','L_Hip',\
                'L_Knee','L_Ankle',\
                'B_Pelvis','B_Spine',\
                'B_Neck','B_Head',\
                'R_Wrist','R_Elbow',\
                'R_Shoulder','L_Shoulder',\
                'L_Elbow','L_Wrist',\
                'L_eye','L_ear',\
                'R_eye','R_ear',\
                'Nose'

    def