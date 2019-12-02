import cv2 as cv

import picamera
from picamera import PiCamera
import picamera.array
from imutils.video.pivideostream import PiVideoStream
from picamera.array import PiRGBArray
import io
import imutils
import time
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array

scaleX = 456
scaleY = 342
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]

# load tensorflow model for emotion recognition
sess = tf.Session()
f = tf.gfile.FastGFile('/home/pi/Documents/tfEmotion/tf_model.pb', 'rb')

graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())
sess.graph.as_default()
g_in = tf.import_graph_def(graph_def)

tensor_output = sess.graph.get_tensor_by_name('import/global_average_pooling2d_1/Mean:0')
tensor_input = sess.graph.get_tensor_by_name('import/input_1:0')

def getTfEmotion(roi):

      label = 'noEmotionDetected'
      if roi.any():
            roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
            roi = cv.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = sess.run(tensor_output,  {tensor_input: roi})
            label = EMOTIONS[preds.argmax()]
      return label
