import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def get_img_metrics(img):
    img_metric = img.copy()
    # make all white pixels black (to make only red and blue left)
    img_metric[cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) == 255] = (0,0,0)
    # red channel lost material
    lost_material = (img_metric[:,:,0] / 255.).astype(int)
    # blue channel gained material
    gained_material = (img_metric[:,:,2] / 255.).astype(int)

    pixelcount = img_metric.shape[0] * img_metric.shape[1]
    prop_lost_px = lost_material.sum() / pixelcount
    prop_gained_px = gained_material.sum() / pixelcount
    prop_changed_px = (prop_lost_px + prop_gained_px)

    return (prop_lost_px, prop_gained_px, prop_changed_px)

def get_directory_metrics(source_dir, dest_dir=None):
    if dest_dir is None:
        dest_dir = source_dir

    items = os.scandir(source_dir)
    itemcount = len(list(os.scandir(source_dir)))
    df_list = []
    for item in tqdm(items, total=itemcount):
        if item.is_dir():
            img = cv2.cvtColor(cv2.imread(os.path.join(source_dir, item.name, "heatmap.tif"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            df_list.append((item.name, *get_img_metrics(img)))

    df = pd.DataFrame(df_list, columns=["serial", "prop_lost_px", "prop_gained_px", "prop_changed_px"])
    df.to_json(os.path.join(dest_dir, "metrics.json"))
    # df.to_pickle(os.path.join(dest_dir, "metrics.pickle"))