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
from networkComputing import getBodyMeasures, getPersonRecog, getEmotion, updateEncodings
from encode_faces import encodeFaces
import numpy
from sendMail import sendMsg
from tfEmotion import getTfEmotion


class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = None
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None
        #self.name = 
        #initialize Warning Variables
        self.previousChestY = 0
        self.intruderArray = numpy.zeros((20))
        self.intruderIndex = 0
        self.emotionArray = numpy.zeros((20))
        self.emotionIndex = 0

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


        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                self.frame = self.vs.read()
                self.frame = cv2.flip(self.frame, 0)
                
                NAME = getPersonRecog(self.frame)
                EMOTION = getTfEmotion(self.frame)
                
                fallWarning = self.fallDetection(self.frame)
                #print (fallWarning)
                
                intruderWarning = self.intruderDetection(self.frame)
                #namess = intruderWarning().name
                #print (intruderWarning)
                #print (self.intruderArray)
                
                emotionWarning = self.emotionDetection(self.frame)
        
                
                # show emotion info on monitor
                cv2.putText(self.frame, emotionWarning, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                # show intruder info on monitor
                cv2.putText(self.frame, intruderWarning, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                # show fall  info on monitor
                cv2.putText(self.frame, fallWarning, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                # show name info on monitor
                
                if NAME[0] == True:
                        cv2.rectangle(self.frame, (int(NAME[2] * 0.658), int(NAME[3] * 0.658)), (int(NAME[4] * 0.658), int(NAME[5] * 0.658)), (0, 255, 0), 1)
                        cv2.putText(self.frame, EMOTION, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(self.frame, NAME[1], (NAME[2], NAME[3]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                #cv2.rectangle(self.frame, (NAME().xMin, NAME().yMin), (NAME().xMax, NAME().yMax), (0,255,0),3)
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
        

    def takeSnapshot(self):

        # grab the current timestamp and use it to construct the
        # output path 

        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        f=open(os.path.join(sys.path[0], 'store.txt'))
        if f.mode == 'r':
            self.outputPath = f.read()
         
        p = os.path.sep.join((self.outputPath, filename))

        #print(self.outputPath)
        #run the encodings:

        
        cv2.imwrite(p, self.frame.copy())
        #cv2.imwrite(p, self.frame)
        print("[INFO] saved {}".format(filename))
        #cv2.putText(self.frame.copy(), 'A new snapshot has been taken', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        encodeFaces()
        updateEncodings()

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.stop()
        self.root.quit()
        
    def fallDetection(self, fframe):
        #Fall detection:
        startTime = time.time()
        chestY, shoulderDiv = getBodyMeasures(fframe)
        chestDiv = chestY - self.previousChestY
        #considering that shoulders are 0.35 meters apart from each other
        #the fall distance in meters can be calculated
        chestDiv = 0.35 * (chestDiv / shoulderDiv)
        endTime = time.time()
        fallRate = chestDiv / (endTime - startTime)
        #print (fallRate)
        self.previousChestY = chestY
        if fallRate > 0.3:
                sendMsg(0)
                return 'FALL DETECTED'
        else:
                return 'no fall detected'

    def intruderDetection(self, fframe):
            personDetected, name , a, b, c, d = getPersonRecog(fframe)
            if self.intruderIndex == 19:
                    self.intruderIndex = 0
                    
            if personDetected == True:
                    if name == 'unknownPerson':
                            self.intruderArray[self.intruderIndex] = 1
                            self.intruderIndex += 1
                            
                    else:
                            #personDetected, name = getPersonRecog(fframe)
                            self.intruderArray[self.intruderIndex] = 0
                            self.intruderIndex += 1
                            #print (name)

            if personDetected == False:
                    self.intruderArray[self.intruderIndex] = 0
                    self.intruderIndex += 1
            
            if numpy.sum(self.intruderArray) > 12:
                    return 'INTRUDER ALERT'
                    
            else:
                    return 'no intruder alert'
                    
    def emotionDetection(self, fframe):
            if self.emotionIndex == 19:
                    self.emotionIndex = 0
            # only return index 0 of function "getEmotion"        
            functionaa = getEmotion()
            self.emotionArray[self.emotionIndex] = functionaa[0]
            #self.emotionArray[self.emotionIndex] = getEmotion()
            
            self.emotionIndex += 1
            
            if numpy.sum(self.emotionArray) > 12:
                    return 'EMOTION ALERT'
                    
            else:
                    return 'no emotion alert'
