# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
import secondwindow
from imutils.video import VideoStream
import time
from secondwindow import Toplevel1
import sys
import re

# initialize the video stream and allow the camera sensor to warmup
#print("[INFO] warming up camera...")
##vs = VideoStream.start(self)
#vs = cv2.VideoCapture(0)
#time.sleep(2.0)


class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        #self.outputPath = outputPath
        #self.outputPath = Toplevel1.lastDir
        self.outputPath = None
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None

        # create a button, that when pressed, will take the current
        # frame and save it to file
        btn = tki.Button(self.root, text="Snapshot!",
            command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,
            pady=10)

        btn1 = tki.Button(self.root, text="Input Your Name First",
            command=self.InputName)
        btn1.pack(side="top", fill="both", expand="yes", padx=10,
            pady=10)

        # start a thread that constantly pools the video sensor for
        # the most recently read frame

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)
        self.Level = None
        self.dirName = ''
    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=300)

                #font = cv2.FONT_HERSHEY_SIMPLEX
                #cv2.putText(img,'OpenCV',(10,500),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255),2,cv2.LINE_AA)
                #add some text

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def InputName(self):

        
        os.system('python secondwindow.py') 
        
    #def GetDirectory(self):
    #    self.Level = Toplevel1()
    #    self.dirName = self.Level.dirName
    #    print(self.Level.dirName)

    def takeSnapshot(self):


        # grab the current timestamp and use it to construct the
        # output path 

        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        #p = os.path.sep.join((self.outputPath, filename))
        f=open('/home/pi/Documents/GUI/store.txt', "r")
        if f.mode == 'r':
            self.outputPath = f.read()
         
        p = os.path.sep.join((self.outputPath, filename))
        #p = secondwindow.Toplevel1.confirm(self).dirName
        #p = Toplevel1.confirm(self).dirName
        # save the file
        #self.GetDirectory()
        #p = self.dirName
        #p = secondwindow.getDir()
        #p = 'C:\\Users\\xiaoc\\Desktop\\chao'
        print(self.outputPath)
        cv2.imwrite(p, self.frame.copy())
        #cv2.imwrite(p, self.frame)
        print("[INFO] saved {}".format(filename))

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()

    