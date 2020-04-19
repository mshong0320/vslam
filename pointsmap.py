#!/usr/bin/env python3
import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import time
import numpy as np
from display import Display
import OpenGL.GL as gl
# import pangolin
import pypangolin as pango
from multiprocessing import Process, Queue
import g2o
from utils import *


class Point:
    # A Point is a 3-D point in the world
    # Each Point is observed in multiple Frames
  def __init__(self, mapp, loc, color):
    self.frames = []
    self.idxs = []
    self.pt = loc
    self.color = np.copy(color)
    self.id = len(mapp.points)
    mapp.points.append(self)

  # where it is on the frame
  def add_observation(self, frame, idx):
    frame.pts[idx] = self
    self.frames.append(frame)
    self.idxs.append(idx)


class Map:
    def __init__(self):
        self.frames = []
        self.points = []
        # self.q = Queue()
        self.q = None
        self.state = None

    # *** optimizer ***

    def optimize(self):
        # create g2o optimizer
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)

        robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

        # add frames to graph
        for f in self.frames:
          pose = f.pose
          #pose = np.linalg.inv(pose)
          sbacam = g2o.SBACam(g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3]))
          sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[0][2], f.K[1][2], 1.0)

          v_se3 = g2o.VertexCam()
          v_se3.set_id(f.id)
          v_se3.set_estimate(sbacam)
          v_se3.set_fixed(f.id <= 1)
          opt.add_vertex(v_se3)

        # add points to frames
        PT_ID_OFFSET = 0x10000
        for p in self.points:
          pt = g2o.VertexSBAPointXYZ()
          pt.set_id(p.id + PT_ID_OFFSET)
          pt.set_estimate(p.pt[0:3])
          pt.set_marginalized(True)
          pt.set_fixed(False)
          opt.add_vertex(pt)

          for f in p.frames:
            edge = g2o.EdgeProjectP2MC()
            edge.set_vertex(0, pt)
            edge.set_vertex(1, opt.vertex(f.id))
            uv = f.kpus[f.pts.index(p)]
            edge.set_measurement(uv)
            edge.set_information(np.eye(2))
            edge.set_robust_kernel(robust_kernel)
            opt.add_edge(edge)
            
        opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(50)

        # put frames back
        for f in self.frames:
          est = opt.vertex(f.id).estimate()
          R = est.rotation().matrix()
          t = est.translation()
          f.pose = poseRt(R, t)

        # put points back
        for p in self.points:
          est = opt.vertex(p.id + PT_ID_OFFSET).estimate()
          p.pt = np.array(est)


    def create_viewer(self):
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()


    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)
        
    def viewer_init(self, w, h):
        pango.CreateWindowAndBind('Main', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        pm = pango.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 1000)
        mv = pango.ModelViewLookAt(0, -25, -50, 0, 0, 0, 0, -1, 0)
        self.scam = pango.OpenGlRenderState(pm, mv)
        ui_width = 180
        self.handler = pango.Handler3D(self.scam)

        self.d_cam = pango.CreateDisplay().SetBounds(pango.Attach(0),
                                                     pango.Attach(1),
                                                     pango.Attach.Pix(ui_width),
                                                     pango.Attach(1),
                                                     -w/h).SetHandler(self.handler)

        # hack to avoid small Pangolin, no idea why it's *2
        self.d_cam.Resize(pango.Viewport(0,0,w*2,h*2))
        self.d_cam.Activate(self.scam)

    def viewer_refresh(self, q):
        #turn state into points
        if self.state is None or not q.empty():
            self.state = q.get()

        # ppts = np.array([d[:3, 3] for d in self.state[0]])
        # spts = np.array(self.state[1])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.d_cam.Activate(self.scam)

        # draw poses
        # colors = np.zeros((len(ppts), 3))
        # colors[:, 0] = 1
        # colors[:, 1] = 1 
        # colors[:, 2] = 0 
        gl.glPointSize(10)
        gl.glColor3f(0.0, 0.0, 1.0)
        # pango.DrawPoints(ppts, colors)
        # pango.DrawPoints(self.state[0], self.state[2])
        pango.DrawCameras(self.state[0])

        # # draw keypoints
        # colors = np.zeros((len(spts), 3))
        # colors[:, 0] = 0
        # colors[:, 1] = 1 
        # colors[:, 2] = 0
        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pango.DrawPoints(self.state[1], self.state[2])
        pango.FinishFrame()

    def display_map(self):
        if self.q is None:
            return
        poses, pts, colors = [], [], []
        for f in self.frames:
            poses.append(f.pose)

        for f in self.points:
            pts.append(f.pt)
            colors.append(f.color)

        self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))


