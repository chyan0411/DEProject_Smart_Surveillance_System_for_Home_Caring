import cv2 as cv

import picamera
from picamera import PiCamera
import picamera.array
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
import io
import imutils
import time
import math

# Load the model.
net = cv.dnn.readNet('human-pose-estimation-0001.xml',
                     'human-pose-estimation-0001.bin')
# Specify target device.
net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

# starting video streaming
cv.namedWindow('pose')
vs = PiVideoStream().start()
time.sleep(2.0)

joint = 0
chestY = 0


while True:
    startTime = time.time()
    frameArray = vs.read()
    captureTime = time.time()
    frameClone = imutils.resize(frameArray, width = 456)
    
    blob = cv.dnn.blobFromImage(frameArray, size=(456, 256), ddepth=cv.CV_8U)
    prepareTime = time.time()
    net.setInput(blob)
    heatmap = net.forward()
    processTime = time.time()
    rightShoulder = False
    leftShoulder = False
    chest = False
    rightShoulderX = 0
    rightShoulderY = 0
    leftShoulderX = 0
    leftShoulderY = 0
    
    for i in range(18):
      v = 0
      x = 0
      y = 0
      for j in range(32):
        for k in range(57):
          if heatmap[0,i,j,k] > v:
            v = heatmap[0,i,j,k]
            x = k
            y = j
      if v > 0.4:
        cv.circle(frameClone,(int(x*8) ,int(y*11)), 5, (0,0,255), -1)
        
        if i == 1:
          prevChestY = chestY
          chestY = y
          chest = True
          
        if i == 2:
          rightShoulderX = x
          rightShoulderY = y
          rightShoulder = True
          
        if i == 5:
          leftShoulderX = x
          leftShoulderY = y
          leftShoulder = True
    
    #calculate the distance of the shoulders in the picture:    
    shoulderDivX = rightShoulderX - leftShoulderX
    shoulderDivY = rightShoulderY - leftShoulderY
    shoulderDiv = math.sqrt(shoulderDivX**2 + shoulderDivY**2)
    #calculate the distance of current and previous chest y-position
    chestDiv = chestY - prevChestY
    #considering that shoulders are 0.35 meters apart from each other
    #the fall distance in meters can be calculated
    chestDiv = 0.35 * (chestDiv / shoulderDiv)
    
    #print (shoulderDiv)
                 
    cv.imshow('pose', frameClone)
      
    showTime = time.time()
    loopDuration =  '%.2f' % (showTime - startTime)
    captureDuration = '%.2f' % (captureTime - startTime)
    prepareDuration = '%.2f' % (prepareTime - captureTime)
    processDuration = '%.2f' % (processTime - prepareTime)
    showDuration = '%.2f' % (showTime - processTime)
    
    #fall rate in meters per second:
    fallRate = chestDiv / (showTime - startTime)
    if fallRate > 0.3 and rightShoulder == True and leftShoulder == True and chest == True:
      print ('fall detected')
    
    with open('/home/pi/Documents/intelPose/Durations2.txt', 'a') as the_file:
      the_file.write('Whole Loop: ' + loopDuration + 's\n'
                              + 'Capture: ' + captureDuration + 's\n'
                              + 'Prepare: ' + prepareDuration + 's\n'
                              + 'Process: ' + processDuration + 's\n'
                              + 'Show: ' + showDuration + 's\n\n')

 

    if cv.waitKey(1) & 0xFF == ord('q'):
      break

                
cv.destroyAllWindows()
