# %%
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.optimize import minimize_scalar

%matplotlib qt

im_template = cv.imread(r'C:\temporary work folder gsk35\UROP tests 2\image0.tif', cv.IMREAD_GRAYSCALE)

im = cv.imread(r'C:\temporary work folder gsk35\UROP tests 2\processed_scan_run4.tif', cv.IMREAD_GRAYSCALE)
# pad image with zeros, make it somewhat bigger so we can account for shift
im = np.pad(im, int(im.shape[0]*0.1), mode='constant', constant_values=0)

# %%

def ccorr(rot: float):
    rows, cols = im_template.shape
    M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), rot, 1)
    im_rot = cv.warpAffine(im_template, M, (cols,rows))
    return cv.matchTemplate(im, im_rot, cv.TM_CCOEFF)

res = minimize_scalar(lambda x: -ccorr(x).max(), bounds=(0,60), method='bounded', options={'disp': True})
result = ccorr(res.x)

# th = np.linspace(0,360,10)
# coeffs = np.zeros_like(th)
# for i in trange(len(th)):
#     coeffs[i] = ccorr(th[i]).max()
# plt.plot(th, coeffs)
# plt.show()

# %%

plt.imshow(result)
plt.show()

# %%

minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
rows, cols = im_template.shape
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), res.x, 1)
im_rot = cv.warpAffine(im_template, M, (cols*2,rows*2))
im_annotate = cv.cvtColor(im, cv.COLOR_GRAY2BGR)

# %%
# put rectangle in place of maxLoc

cv.rectangle(im_annotate, maxLoc, (maxLoc[0]+cols, maxLoc[1]+rows), (255,0,0), 5)
im_annotate[np.nonzero(im_rot!=0)] = np.array([0,255,0])
fig, ax = plt.subplots(1,2)
ax[0].imshow(im_rot, cmap='gray')
ax[1].imshow(im_annotate)
plt.show()

# %%