# import the necessary packages
from __future__ import print_function
from btn import PhotoBoothApp
from imutils.video import VideoStream
import imutils
import argparse
import time
import tkinter
'''
# from secondwindow import confirm
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=False,
	help="path to output directory to store snapshots")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
'''
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera = True).start()
time.sleep(2.0)

# start the app

pba = PhotoBoothApp(vs, '')
pba.root.mainloop()
