import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('D:/0801.png')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = np.array(image,dtype='float32')

print(image.shape)


