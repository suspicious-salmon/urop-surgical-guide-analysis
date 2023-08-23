import cv2
import numpy as np
import matplotlib.pyplot as plt

import utility as u

def heatmap(img_cad_dir, img_scan_dir, out_dir):
    img_cad = u.readim(img_cad_dir, cv2.IMREAD_GRAYSCALE)
    img_scan = u.readim(img_scan_dir, cv2.IMREAD_GRAYSCALE)

    # shouldn't be necessary, input images should already be binary
    img_cad = cv2.threshold(img_cad, 20, 255, cv2.THRESH_BINARY)[1]
    img_scan = cv2.threshold(img_scan, 20, 255, cv2.THRESH_BINARY)[1]

    assert not np.any(np.logical_and(img_cad != 0, img_cad != 255)), "images must be purely binary, 0 or 255"
    assert not np.any(np.logical_and(img_scan != 0, img_scan != 255)), "images must be purely binary, 0 or 255"

    img_lost_material = np.logical_and(img_cad == 255, img_scan == 0).astype(int)
    img_gained_material = np.logical_and(img_cad == 0, img_scan == 255).astype(int)

    heatmap = cv2.cvtColor(255-img_cad, cv2.COLOR_GRAY2RGB)
    heatmap[img_lost_material == 1] = (0,0,255)
    heatmap[img_gained_material == 1] = (255,0,0)

    cv2.imwrite(out_dir, heatmap)