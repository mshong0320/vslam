#!/usr/bin/env python3
import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
from utils import extract, normalize


class Frame:
    def __init__(self, mapp, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = np.eye(4)
        self.h, self.w = img.shape[0:2]

        self.kpus, self.des = extract(img)
        self.kps = normalize(self.Kinv, self.kpus)
        self.pts = [None]*len(self.kps)

        self.id = len(mapp.frames)
        mapp.frames.append(self)