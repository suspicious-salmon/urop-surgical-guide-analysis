import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

import utility as u

cad_dir = "image0.tif"
cad_img = u.readim(cad_dir, cv2.IMREAD_GRAYSCALE)

scan_dir = "mxd_scan017_part_10.tif"
scan_img = u.readim(scan_dir, cv2.IMREAD_COLOR)

# scan_img_blue = np.zeros((cad_img.shape[0], cad_img.shape[1], 3), np.uint8)
# scan_img_blue[:,:,0] = scan_img

processed_scanr4_dir = "processed_scan_run4.tif"
processed_scanr4_img = u.readim(processed_scanr4_dir, cv2.IMREAD_COLOR)
processed_scanr4_img[np.where(processed_scanr4_img[:,:,0] == 255)] = (0,255,0)

processed_scanr1_dir = "processed_scan_run1.tif"
processed_scanr1_img = u.readim(processed_scanr1_dir, cv2.IMREAD_COLOR)
cv2.addWeighted(scan_img, 0.9, processed_scanr4_img, 0.1, 0)

display = cv2.addWeighted(scan_img, 0.9, processed_scanr4_img, 0.1, 0)
ax = plt.subplot(121)
ax.set_title("Run 4")
ax.imshow(display)

display = cv2.addWeighted(scan_img, 0.9, processed_scanr1_img, 0.1, 0)
ax = plt.subplot(122)
ax.set_title("Run 1")
ax.imshow(display)

plt.show()