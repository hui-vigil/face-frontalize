import cv2
import mtcnn
import numpy as np


img_path = r'E:\file\face-generation\AFF-GAN\mask128.png'
img = cv2.imread(img_path, 0)

for i in range(128):
    for j in range(128):
        print(img[i, j], end='')
    print()