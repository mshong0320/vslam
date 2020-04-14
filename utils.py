import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import time
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform
 

def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

#pose estimation: Extract cameras. Decompose E -> SR where S is skew symm matrix and R is a rotation.
def extractRot_trans(E):
    W = np.mat([[0, -1, 0], [1,0,0], [0,0,1]], dtype=float) # rotation
    Z = np.mat([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], dtype=float) # skew symmetric
    # dot(Z,W) = diag([1, 1, 0])
    # dot(Z,W.T) = -diag([1, 1, 0])
    # if E has the SVD we can find two solutions; E = S1R1, where S1 = -UZU.T, R1 = UW.TV.T, and 
    # E = S2R1, wheree S2 = UZU.T, R2 = UWV.T
    # to see that these are valid solutions: R1.T*R2 = (UW.T*V.T).T*U*W.T*V.T = V*W*U.T*U*W.T*V.T = I

    U,d,Vt = np.linalg.svd(E)
    assert np.linalg.det(U)>0

    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)

    # !!!!! very import part!!! 
    if np.sum(R.diagonal()) < 0: # if det(UV.T) = -1
        R = np.dot(np.dot(U, W.T), Vt)

    t = U[:,2]
    SE3 = np.eye(4)
    SE3[:3, :3] = R
    SE3[:3, 3] = t
    return SE3

def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
    ret = np.dot(K, np.array([pt[0],pt[1],1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

def extract(img):
    orb = cv2.ORB_create()
    #detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
    #extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
    kps, des = orb.compute(img, kps)
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des

def match_frames(f1, f2):
    #matching
    ret = [] 
    idx1, idx2 = [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)
    
    #Lowe's ratio test
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            #keep around indices into the frame
            idx1.append(m.queryIdx)
            idx2.append(m.trainIdx)

            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            ret.append((p1, p2))

    assert len(ret) >= 8
    # print(len(ret))
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    #filter
    model, inliers = ransac((ret[:,0], ret[:,1]),
                             EssentialMatrixTransform, 
                             # FundamentalMatrixTransform,
                             min_samples=8, 
                             residual_threshold=0.005, 
                             max_trials=200)

    # model.params -> 3x3 fundamental matrix

    #ignore outliers
    ret = ret[inliers]
    SE3 = extractRot_trans(model.params)

    print(sum(inliers), len(inliers))
    
    # index intot he array
    return idx1[inliers], idx2[inliers], SE3
