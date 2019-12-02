import cv2 as cv

import pickle
import scipy.spatial
import io
import imutils
import time
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from tfEmotion import getTfEmotion
import math

# Load the model.
faceNet = cv.dnn.readNet('face-detection-retail-0004.xml',
                     'face-detection-retail-0004.bin')
poseNet = cv.dnn.readNet('human-pose-estimation-0001.xml',
                     'human-pose-estimation-0001.bin')
reidentNet = cv.dnn.readNet('face-reidentification-retail-0095.xml',
                     'face-reidentification-retail-0095.bin')
# Specify target device.
faceNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
poseNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
reidentNet.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)

#load the encoded faces
data = pickle.loads(open('encodings.pickle', "rb").read())
detectionThreshold = 0.6


scaleX = 456
scaleY = 342
badEmotions = ["angry" ,"disgust","scared", "sad"]
#Emotions = ["angry" ,"disgust","scared", "sad", "surprised", "happy", "neutral"]
faces = 0
roi = None
xMin = 0 
yMin = 0 
xMax = 0
yMax = 0


def getBodyMeasures (uiFrame):
      
      frameClone = imutils.resize(uiFrame, width = 456)
      blob = cv.dnn.blobFromImage(uiFrame, size=(456, 256), ddepth=cv.CV_8U)
      poseNet.setInput(blob)
      heatmap = poseNet.forward()
      
      #calculating chest Y value and shoulder difference:
      rightShoulder = False
      leftShoulder = False
      chest = False
      
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
                  if i == 1:
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
      if chest == True and rightShoulder == True and leftShoulder == True:
            shoulderDivX = rightShoulderX - leftShoulderX
            shoulderDivY = rightShoulderY - leftShoulderY
            shoulderDiv = math.sqrt(shoulderDivX**2 + shoulderDivY**2)
            return chestY, shoulderDiv
      else:
            return 99,99

def getPersonRecog (uiFrame):
      global faces
      global roi
      global xMin , yMin, xMax, yMax
      #first detect if there is a face:
      data = pickle.loads(open('encodings.pickle', "rb").read())
      frameClone = imutils.resize(uiFrame, width = 456)
      blob = cv.dnn.blobFromImage(uiFrame, size=(300, 300), ddepth=cv.CV_8U)
      faceNet.setInput(blob)
      outputBlob = faceNet.forward()
      
      faces = 0
      minDistance = 1
      index = 0
      detectionThreshold = 0.6
      
      for i in range(200):
            if outputBlob[0,0,i,2] > 0.3:
                  xMin = int(outputBlob[0,0,i,3] * scaleX)
                  yMin = int(outputBlob[0,0,i,4] * scaleY)
                  xMax = int(outputBlob[0,0,i,5] * scaleX)
                  yMax = int(outputBlob[0,0,i,6] * scaleY)
                  faces += 1
      
      #if there is a face, compare it to the ones in the dataset:
      if faces > 0:
            roi = frameClone[yMin:yMax, xMin:xMax]
            if roi.any():
                  # compute the 256 value array for the detected face:
                  reidentBlob = cv.dnn.blobFromImage(roi, size=(128, 128), ddepth=cv.CV_8U)
                  reidentNet.setInput(reidentBlob)
                  reidentOutputBlob = reidentNet.forward()
                  
                  #compare the array to the ones in the database
                  for x in range (len(data["encodings"])-1):
                        distance = scipy.spatial.distance.cosine(reidentOutputBlob, data["encodings"][x])
                        if distance < minDistance:
                              minDistance = distance
                              index = x
                              
            if minDistance < detectionThreshold:
                  name = data["names"][index]
                  return True, name , xMin, yMin, xMax, yMax
            else:
                  return True, 'unknownPerson', xMin , yMin, xMax, yMax
      else:
            return False, 'noFaceDetected', xMin , yMin, xMax, yMax

def getEmotion ():
      global faces
      global roi
      
      if faces > 0:
            emotion = getTfEmotion(roi)
            if emotion in badEmotions:
                  return 1, emotion
            else:
                  return 0,emotion 
      else:
            return 0, "no face"

def updateEncodings():
      data = pickle.loads(open('encodings.pickle', "rb").read())
