"""This script goes through all the processed images & heatmaps of the surgical guides.
On the left will be shown the original photo overlaid with the segmented outline in green.
On the right will be the part's heatmap.
By default the heatmaps will be shown in descending order of changed pixels (sum of no. of lost and no. of gained pixels from cad img to scan img)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

import _cvutil

root_results_folder = r"E:\greg\Organised Code\EnvTests\Results\Run1"

root_dataset_folder = r"E:\greg\Dogs\UIUCMxD"
cad_folder = os.path.join(root_dataset_folder, "CAD")
images_folder = os.path.join(root_dataset_folder, "Images")
csv_directory = os.path.join(root_dataset_folder, "mxd_258key.csv")
parts_df = pd.read_csv(csv_directory)

# some serials that had interesting heatmaps
SERIALS_LIST = [
    "GUIDE-0571-0E03",
    "GUIDE-0151-01EV"
    "GUIDE-1156-1523",
    "GUIDE-1171-1540"
]

metrics_df = pd.read_json(os.path.join(root_results_folder, "metrics.json"))
sorted_metrics_df = metrics_df.sort_values("prop_changed_px", ascending=False, ignore_index=True)
# sorted_metrics_df = metrics_df.sort_values("serial", ascending=True, ignore_index=True)

# iterate over sorted_metrics_df
for count, row in tqdm(sorted_metrics_df.iterrows(), total=metrics_df.shape[0]):

    # if row["serial"] in SERIALS_LIST: # uncomment this line to only see the interesting heatmaps

        print(row["serial"])
        print(parts_df.loc[parts_df["serial"] == row["serial"]]["img_name"])
        scan = _cvutil.readim(os.path.join(root_results_folder, "aligned_scans", row["serial"] + "_aligned.tif"), cv2.IMREAD_COLOR)
        processed_scan = _cvutil.readim(os.path.join(root_results_folder, "steps", row["serial"] + "_processed.tif"), cv2.IMREAD_COLOR)
        heatmap = cv2.cvtColor(_cvutil.readim(os.path.join(root_results_folder, "heatmaps", row["serial"] + "_heatmap.tif"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # convert grayscale to green
        processed_scan[np.where(processed_scan[:,:,0] == 255)] = (0,255,0)

        plt.subplot(121), plt.title("Scan overlaid with segmented lamina in green")
        plt.imshow(cv2.addWeighted(scan, 0.9, processed_scan, 0.1, 0))

        plt.subplot(122), plt.title(f"Heatmap: blue gained material, red lost. \n Changed px: {(100*row['prop_changed_px']):.2f}%")
        plt.imshow(heatmap)

        plt.suptitle(row["serial"])
        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()