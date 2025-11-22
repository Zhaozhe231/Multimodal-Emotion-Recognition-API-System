"Author: Zhaozhe Zhang"


import cv2
from picamera2 import Picamera2
import time


picam2 = Picamera2()
config = picam2.create_still_configuration()
picam2.configure(config)


picam2.start()
time.sleep(2)


filepath = '/home/Evan/photo1.jpg'
picam2.capture_file(filepath)
picam2.stop()
img = cv2.imread(filepath)
flipped = cv2.flip(img, -1)
# file store path
cv2.imwrite('/home/Evan/emotion_detection/Facial-Emotion-Recognition-with-OpenCV-and-Deepface/images/photo2.jpg', flipped)
