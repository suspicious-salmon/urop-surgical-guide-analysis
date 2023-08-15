import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

# img_dir = r"C:\temporary work folder gsk35\Production\Run1\GUIDE-0001-0000\heatmap.tif"
# img = cv2.imread(img_dir, cv2.IMREAD_COLOR)

def get_metrics(img):
    imgtemp = img.copy()
    imgtemp[cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) == 255] = (0,0,0)
    # red channel lost material
    lost_material = (imgtemp[:,:,0] / 255.).astype(int)
    # blue channel gained material
    gained_material = (imgtemp[:,:,2] / 255.).astype(int)


    # plt.subplot(131), plt.imshow(img)
    # plt.subplot(132), plt.imshow(lost_material, cmap="gray")
    # plt.subplot(133), plt.imshow(gained_material, cmap="gray")
    # plt.show()

    # plt.subplot(121), plt.hist(lost_material.flatten(), bins=20)
    # plt.subplot(122), plt.hist(gained_material.flatten(), bins=20)
    # plt.show()

    pixelcount = imgtemp.shape[0] * imgtemp.shape[1]
    prop_lost_px = lost_material.sum() / pixelcount
    prop_gained_px = gained_material.sum() / pixelcount
    prop_changed_px = prop_lost_px + prop_gained_px / pixelcount

    return (prop_lost_px, prop_gained_px, prop_changed_px)

source_dir = r"C:\temporary work folder gsk35\Production\Run1" # directory of folder containing an analysis run
items = os.scandir(source_dir)
itemslen = os.scandir(source_dir)
df_list = []
for item in tqdm(items, total=len(list(itemslen))):
    if item.is_dir():
        img = cv2.imread(os.path.join(source_dir, item.name, "heatmap.tif"), cv2.IMREAD_COLOR)
        df_list.append((item.name, *get_metrics(img)))

df = pd.DataFrame(df_list, columns=["serial", "prop_lost_px", "prop_gained_px", "prop_changed_px"])
print(df)
df.to_json("metrics.json")
df.to_pickle("metrics.pickle")