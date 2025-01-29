import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import time

cwd = os.getcwd()

for image in os.listdir(cwd + "\\dataset\\taken_images"):
    image_path = os.path.join(cwd + "\\dataset\\taken_images\\", image)
    img = cv2.imread(image_path)
    img = img[110:712 ,180:1000]
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img, (224, 224))
    #_, img = cv2.threshold(img, 90 , 255, cv2.THRESH_BINARY)  # Change 127 (threshold value)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    break

print(cwd)
img = cv2.imread(cwd + "\\dataset\\image_207.jpg")
img = img[110:712 ,180:1000]
cv2.imshow("Image", img)
cv2.waitKey(0)