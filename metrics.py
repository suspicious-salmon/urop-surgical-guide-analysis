import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import utility as u

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

def get_directory_metrics(source_folder, dest_folder):
    df_list = []
    for file in tqdm(os.scandir(source_folder), total=len(list(os.scandir(source_folder)))):
        heatmap = cv2.cvtColor(u.readim(os.path.join(source_folder, file.name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        df_list.append((file.name.split("_")[0], *get_img_metrics(heatmap)))

    df = pd.DataFrame(df_list, columns=["serial", "prop_lost_px", "prop_gained_px", "prop_changed_px"])
    if not os.path.isfile(os.path.join(dest_folder, "metrics.json")):
        df.to_json(os.path.join(dest_folder, "metrics.json"))
    else:
        print("WARNING: a metrics file already exists. Metrics will be saved in metrics_overwrite.json. It is at risk of being overwritten next time this code is run.")
        df.to_json(os.path.join(dest_folder, "metrics_overwrite.json"))
        
    # df.to_pickle(os.path.join(dest_dir, "metrics.pickle"))