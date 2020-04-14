import os
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import sdl2
import sdl2.ext
import time
import numpy as np


class Display:
    def __init__(self, W, H):
        sdl2.ext.init()
        self.window = sdl2.ext.Window("twitch SLAM", size=(W,H))
        self.window.show()
        self.W = W
        self.H = H

    def paint(self, img):
        img = cv2.resize(img, (self.W, self.H))
        events = sdl2.ext.get_events()

        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        #draw
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:,:,0:3] = img.swapaxes(0,1)
        
        #blit
        self.window.refresh()