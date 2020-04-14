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

W = 1920//2
H = 1080//2
F = 270
K = np.array([[F,0,W//2],
              [0,F,H//2],
              [0,0,   1]]) 


class Map:
    def __init__(self):
        self.frames = []
        self.points = []
        self.q = Queue()
        self.state = None
        p = Process(target=self.viewer_thread, args=(self.q,))
        #start the process
        p.daemon = True
        p.start()

    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)
        
    def viewer_init(self, w, h):
        pango.CreateWindowAndBind('main', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        pm = pango.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 1000)
        mv = pango.ModelViewLookAt(0, -20, -20, 0, 0, 0, 0, -1, 0)
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
        # pango.CreatePanel("ui").SetBounds( pango.Attach(0),
        #                                    pango.Attach(1),
        #                                    pango.Attach(0),
        #                                    pango.Attach.Pix(ui_width))

    def viewer_refresh(self, q):
        #turn state into points
        if self.state is None or not q.empty():
            self.state = q.get()

        ppts = np.array([d[:3, 3] for d in self.state[0]])
        spts = np.array(self.state[1])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.d_cam.Activate(self.scam)

        # draw poses
        colors = np.zeros((len(ppts), 3))
        colors[:, 0] = 1
        colors[:, 1] = 1 
        colors[:, 2] = 0 
        gl.glPointSize(10)
        gl.glColor3f(0.0, 0.0, 1.0)
        pango.DrawPoints(ppts, colors)

        # print("here")
        # print(ppts)

        # print("222")
        # print(self.state[0][:-1])
        pango.DrawCameras(self.state[0])

        # # draw keypoints
        colors = np.zeros((len(spts), 3))
        colors[:, 0] = 0
        colors[:, 1] = 1 
        colors[:, 2] = 0
        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pango.DrawPoints(spts, colors)
        pango.FinishFrame()

    def display_map(self):
        if self.q is None:
            return
        poses, pts = [], []
        for f in self.frames:
            poses.append(f.pose)

        for f in self.points:
            pts.append(f.pt)

        self.q.put((poses, pts))



class Point:
  def __init__(self, mapp, location):
    self.frames = []
    self.idxs = []
    self.pt = location
    self.id = len(mapp.points)
    mapp.points.append(self)

  # where it is on the frame
  def add_observation(self, frame, idx):
    self.frames.append(frame)
    self.idxs.append(idx)