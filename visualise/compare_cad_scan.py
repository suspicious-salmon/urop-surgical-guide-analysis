import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

import utility as u

cad_dirs = [
    r"C:\temporary work folder gsk35\UROP tests 2\f\s1\image0.tif",
    r"C:\temporary work folder gsk35\UROP tests 2\f\s2\image0.tif",
    r"C:\temporary work folder gsk35\UROP tests 2\f\s3\image0.tif"
]

scan_dirs = [
    r"C:\temporary work folder gsk35\UIUCMxD\Images\mxd_scan013_part_03.tif",
    r"C:\temporary work folder gsk35\UIUCMxD\Images\mxd_scan056_part_17.tif",
    r"C:\temporary work folder gsk35\UIUCMxD\Images\mxd_scan065_part_01.tif"
]

Ms = [
    np.array([[0.890816242939092, -0.45436375440376, 1346.9382394643744], [0.45436375440376, 0.890816242939092, -835.6840984908605]]),
    np.array([[0.809714328895865, -0.586824254424371, 2346.3698879925787], [0.586824254424371, 0.809714328895865, -1259.1894163755671]]),
    np.array([[0.86555927679745, -0.5008064879271, 1653.5431347931583], [0.5008064879271, 0.86555927679745, -1010.0961596535967]])
]

for i in range(len(cad_dirs)):

    cad_dir = cad_dirs[i]
    img_cad = u.readim(cad_dir, cv2.IMREAD_GRAYSCALE)

    scan_dir = scan_dirs[i]
    img_scan = u.readim(scan_dir, cv2.IMREAD_COLOR)
    M = Ms[i]
    img_scan = cv2.warpAffine(img_scan, M, (img_scan.shape[1], img_scan.shape[0]))

    # aligned_dir = r"C:\temporary work folder gsk35\UROP tests 2\f\out1\Aligned 2 of 20001.tif"
    # img_aligned = u.readim(aligned_dir, cv2.IMREAD_COLOR)

    img_scan[img_cad > 200] = (100,100,100)

    im1 = img_scan.copy()
    im1[img_cad != 0] = (0,0,0)

    im2 = img_cad.copy()
    im2[cv2.cvtColor(img_scan, cv2.COLOR_BGR2GRAY) > 70] = 0
    # img_aligned[img_cad > 200] = (100,100,100)

    ax = plt.subplot(121)
    ax.set_title("im1")
    ax.imshow(im1)

    ax = plt.subplot(122)
    ax.set_title("im2")
    ax.imshow(im2)

    # ax = plt.subplot(133)
    # ax.set_title("scan")
    # ax.imshow(img_scan)
    # plt.subplot(122), plt.imshow(img_aligned)
    plt.show()