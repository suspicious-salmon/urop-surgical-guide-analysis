import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import utility as u

cad_dir = r"E:\greg\Results\Run4\GUIDE-0451-0550\GUIDE-0451-0550_image0.tif"
img_dir = r"E:\greg\Results\Run4\GUIDE-0451-0550\aligned\GUIDE-0451-0550_Aligned 2 of 20001.tif"
# out_dir = r"E:\greg\Results\Run4\GUIDE-0451-0550\aligned\hh.tif"

cad_img = u.readim(cad_dir, cv2.IMREAD_GRAYSCALE)
img = u.readim(img_dir, cv2.IMREAD_GRAYSCALE)

# cad_img_inflated = cv2.resize(cad_img, (int(math.ceil(cad_img.shape[1]*inflation)), int(math.ceil(cad_img.shape[0]*inflation))), interpolation=cv2.INTER_NEAREST)
# cad_img_inflated = cad_img_inflated[math.floor((cad_img_inflated.shape[0]-cad_img.shape[0])/2):cad_img_inflated.shape[0] - math.ceil((cad_img_inflated.shape[0]-cad_img.shape[0])/2),
#                                     math.floor((cad_img_inflated.shape[1]-cad_img.shape[1])/2):cad_img_inflated.shape[1] - math.ceil((cad_img_inflated.shape[1]-cad_img.shape[1])/2)]
# print(cad_img_inflated.shape, cad_img.shape)

kernel_size = 300
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
cad_img_inflated = cv2.dilate(cad_img, kernel)
img[cad_img_inflated == 0] = 100

plt.imshow(img)
plt.show()

# cv2.imwrite(out_dir, img)