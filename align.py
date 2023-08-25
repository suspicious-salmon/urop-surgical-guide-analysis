# %%
import cv2
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.optimize import minimize_scalar
from multiprocessing import Pool

import utility as u

# cad_dir = "E:\greg\Results\Run4\GUIDE-0001-0000\GUIDE-0001-0000_image0.tif"
# scan_dir = "E:\greg\Results\Run4\GUIDE-0001-0000\GUIDE-0001-0000_image1.tif"
# output_folder = "E:\greg\Results\Aligned1"

def align_ccorr(cad_dir, scan_dir, output_dirs):
    cad_img = u.readim(cad_dir, cv2.IMREAD_GRAYSCALE)
    scan_img = u.readim(scan_dir, cv2.IMREAD_GRAYSCALE)

    diameter = math.ceil(math.sqrt(cad_img.shape[0]**2 + cad_img.shape[1]**2))

    cad_img_padded = np.zeros((diameter, diameter), dtype = cad_img.dtype)
    cad_shift_rows = int((cad_img_padded.shape[0] - cad_img.shape[0])/2)
    cad_shift_cols = int((cad_img_padded.shape[1] - cad_img.shape[1])/2)
    cad_img_padded[cad_shift_rows:cad_shift_rows+cad_img.shape[0], cad_shift_cols:cad_shift_cols+cad_img.shape[1]] = cad_img

    scan_img_padded = np.zeros((diameter, diameter), dtype = scan_img.dtype)
    scan_shift_rows = int((scan_img_padded.shape[0] - scan_img.shape[0])/2)
    scan_shift_cols = int((scan_img_padded.shape[1] - scan_img.shape[1])/2)
    scan_img_padded[scan_shift_rows:scan_shift_rows+scan_img.shape[0], scan_shift_cols:scan_shift_cols+scan_img.shape[1]] = scan_img

    def ccorr(rot: float):
        # rotate cad_img_padded by rot degrees
        rows, cols = scan_img_padded.shape
        M = cv2.getRotationMatrix2D(center=((cols-1)/2.0,(rows-1)/2.0), angle=rot, scale=1)
        im_rot = cv2.warpAffine(scan_img_padded, M, dsize=(cols,rows))
        # crop the template
        im_rot = im_rot[scan_shift_rows:scan_shift_rows+scan_img.shape[0], scan_shift_cols:scan_shift_cols+scan_img.shape[1]]

        result = cv2.matchTemplate(cad_img_padded, im_rot, cv2.TM_CCOEFF)
        return result

    res = minimize_scalar(lambda x: -ccorr(x).max(), bounds=(-60,0), method='bounded')
    result = ccorr(res.x)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    rows, cols = cad_img_padded.shape
    M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), res.x, 1)
    scan_img_aligned = cv2.warpAffine(scan_img_padded, M, (scan_img_padded.shape[1], scan_img_padded.shape[0]))
    scan_img_aligned = np.roll(scan_img_aligned, maxLoc[1] - scan_shift_rows, axis=0)
    scan_img_aligned = np.roll(scan_img_aligned, maxLoc[0] - scan_shift_cols, axis=1)

    scan_out = scan_img_aligned[cad_shift_rows:cad_shift_rows+cad_img.shape[0], cad_shift_cols:cad_shift_cols+cad_img.shape[1]]

    # crop out unimportant bits
    kernel_size = 300
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cad_img_inflated = cv2.dilate(cad_img, kernel)
    scan_out[cad_img_inflated == 0] = 0

    u.writeim(output_dirs["aligned"], scan_out)

    extra_pixels = scan_out.copy()
    extra_pixels[cad_img != 0] = 0
    u.writeim(output_dirs["extra_pixels"], extra_pixels)

    missing_pixels = cad_img.copy()
    missing_pixels[scan_out > 20] = 0
    u.writeim(output_dirs["missing_pixels"], missing_pixels)

def crop_to_inflated_cad(cad_dir, img_dir, out_dir, kernel_size=300, overwrite=False):
    cad_img = u.readim(cad_dir, cv2.IMREAD_GRAYSCALE)
    img = u.readim(img_dir, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cad_img_inflated = cv2.dilate(cad_img, kernel)
    img[cad_img_inflated == 0] = 0

    # plt.imshow(img)
    # plt.show()

    u.writeim(out_dir, img, overwrite)

# %%
# def main():
#     df = pd.read_csv('./dataset/mxd_258key.csv', comment='#')
#     cad_dirs = [f'./dataset/CAD/CAD_{x}.tif' for x in df['serial'].tolist()]
#     scan_dirs = [f"./dataset/Images/{os.path.splitext(x)[0]}.jpg" for x in df['img_name'].tolist()]
#     args = [(cad_dirs[i], scan_dirs[i]) for i in range(len(cad_dirs))]
#     print('Starting...')
#     with Pool(8) as p:
#         # p.map(analyse_one, df.itertuples())
#         p.starmap(analyse_one, args)
#     print('\nDone')