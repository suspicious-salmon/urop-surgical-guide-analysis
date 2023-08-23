# %%
import cv2 as cv
from typing import NamedTuple
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from scipy.optimize import minimize_scalar
from multiprocessing import Pool
# %%
def analyse_one(cad_name: str, im_name: str):
    print(f'\rProcessing {cad_name} and {im_name}...', end=' ')
    # pad image with zeros, make it somewhat bigger so we can account for shift
    # pad everything to 7960x7960

    _im_template = cv.imread(cad_name, cv.IMREAD_GRAYSCALE)
    
    # plt.subplot(1,2,1)
    # plt.imshow(_im_template)
    
    im_template = np.zeros((7980, 7980), dtype=_im_template.dtype)
    template_shift_rows = int((im_template.shape[0] - _im_template.shape[0])/2)
    template_shift_cols = int((im_template.shape[1] - _im_template.shape[1])/2)
    im_template[template_shift_rows:template_shift_rows+_im_template.shape[0], template_shift_cols:template_shift_cols+_im_template.shape[1]] = _im_template
    # plt.subplot(1,2,2)
    # plt.imshow(im_template)
    # plt.show()

    _im = cv.imread(im_name, cv.IMREAD_GRAYSCALE)
    # plt.subplot(1,2,1)
    # plt.imshow(_im)
    
    im = np.zeros((7960, 7960), dtype=_im.dtype)
    im_shift_rows = int((im.shape[0] - _im.shape[0])/2)
    im_shift_cols = int((im.shape[1] - _im.shape[1])/2)
    im[im_shift_rows:im_shift_rows+_im.shape[0], im_shift_cols:im_shift_cols+_im.shape[1]] = _im
    # plt.subplot(1,2,2)
    # plt.imshow(im)
    # plt.show()
    # %
    # look for im_template in im
    # use cv.matchTemplate
    def ccorr(rot: float):
        # rotate im_template by rot degrees
        rows, cols = im_template.shape
        M = cv.getRotationMatrix2D(center=((cols-1)/2.0,(rows-1)/2.0), angle=rot, scale=1)
        im_rot = cv.warpAffine(im_template, M, dsize=(cols,rows))
        # crop the template
        im_rot = im_rot[template_shift_rows:template_shift_rows+_im_template.shape[0], template_shift_cols:template_shift_cols+_im_template.shape[1]]

        result = cv.matchTemplate(im, im_rot, cv.TM_CCOEFF)
        return result
    # 
    res = minimize_scalar(lambda x: -ccorr(x).max(), bounds=(0,60), method='bounded')

    result = ccorr(res.x)
    # plt.imshow(result)
    # plt.show()
    # %
    minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(result)
    rows, cols = im_template.shape
    M = cv.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), res.x, 1)
    im_matched = cv.warpAffine(im_template, M, (im_template.shape[1], im_template.shape[0]))
    im_matched = np.roll(im_matched, maxLoc[1] - template_shift_rows, axis=0)
    im_matched = np.roll(im_matched, maxLoc[0] - template_shift_cols, axis=1)

    # plt.imshow(im_matched)
    # plt.show()
    cv.imwrite(f'./dataset/CAD_aligned/{os.path.basename(cad_name)}', im_matched)
    # plot aligned images
    # %
    extra_pixels = im.copy()
    extra_pixels[im_matched!=0] = 0
    cv.imwrite(f'./dataset/extra_pixels/{os.path.basename(im_name)}', extra_pixels)
    # % missing pixels
    missing_pixels = im_matched.copy()
    missing_pixels[im>20] = 0
    cv.imwrite(f'./dataset/missing_pixels/{os.path.basename(im_name)}', missing_pixels)
    print('Done', end=' ')
# %%
def main():
    df = pd.read_csv('./dataset/mxd_258key.csv', comment='#')
    cad_names = [f'./dataset/CAD/CAD_{x}.tif' for x in df['serial'].tolist()]
    im_names = [f"./dataset/Images/{os.path.splitext(x)[0]}.jpg" for x in df['img_name'].tolist()]
    args = [(cad_names[i], im_names[i]) for i in range(len(cad_names))]
    print('Starting...')
    with Pool(8) as p:
        # p.map(analyse_one, df.itertuples())
        p.starmap(analyse_one, args)
    print('\nDone')

if __name__ == '__main__':
    main()