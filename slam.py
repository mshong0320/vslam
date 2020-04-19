#!/usr/bin/env python3
import os
import sys
import numpy.core.multiarray
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import argparse
import sdl2
import sdl2.ext
import time
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
import g2o
from utils import *
from display import Display
from pointsmap import Point, Map
from frame import Frame

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",type=str, default="test_countryroad.mp4")
    args = parser.parse_args()
    return args


# set this!
F = int(os.getenv("F", "800"))
print("Focal length!:")
print(F)
#Camera Intrinsics
W = 1920//2
H = 1080//2
# F = 270
K = np.array([[F, 0, W//2],
              [0, F, H//2],
              [0, 0,    1]]) 
Kinv = np.linalg.inv(K)


mapp = Map()
if os.getenv("D3D") is not None:
  mapp.create_viewer()

display = None
if os.getenv("D2D") is not None:
  display = Display(W, H)

# display = Display(W, H) if os.getenv("D2D") is not None else None

def process_frame(img):
    img = cv2.resize(img, (W,H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return
    print("\n*** frame %d ***" % (frame.id,))

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, SE3 = match_frames(f1, f2)
    f1.pose = np.dot(SE3, f2.pose)

    for i,idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])


    #homogeneous 3-D coords:
    pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
    pts4d /= pts4d[:, 3:]

    
    #reject points without enough parallax
    #reject points behidn the camera
    unmatched_points = np.array([f1.pts[i] is None for i in idx1])
    print("Adding:  %d points" % np.sum(unmatched_points))
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points

    for i,p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        u,v = int(round(f1.kpus[idx1[i], 0])), int(round(f1.kpus[idx1[i], 1]))
        pt = Point(mapp, p, img[v, u])
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])


    for pt1, pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2) 
        cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    # 2-D display
    if display is not None:
        display.paint(img)

    # optimize the map
    if frame.id >= 4:
        mapp.optimize()

    # 3-D display
    mapp.display_map()


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("%s <video.mp4>" % sys.argv[0])
    #     exit(-1)
    args = parse_args()
    vid = args.video
    cap = cv2.VideoCapture(vid)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break