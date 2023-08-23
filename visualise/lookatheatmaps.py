import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

output_folder = r"E:\greg\Results\Run4"
cad_folder = r"C:\temporary work folder gsk35\UIUCMxD\CAD"
images_folder = r"C:\temporary work folder gsk35\UIUCMxD\Images"

parts_csv_dir = r"C:\temporary work folder gsk35\UIUCMxD\mxd_258key.csv"
parts_df = pd.read_csv(parts_csv_dir)

SERIALS_LIST = [
    "GUIDE-0571-0E03",
    "GUIDE-0151-01EV"
]

metrics_df = pd.read_json(os.path.join(output_folder, "metrics.json"))
# sorted_metrics_df = metrics_df.sort_values("prop_changed_px", ascending=False, ignore_index=True)
sorted_metrics_df = metrics_df.sort_values("serial", ascending=True, ignore_index=True)

print(sorted_metrics_df)

for count, row in tqdm(sorted_metrics_df.iterrows(), total=metrics_df.shape[0]):
    # if row["serial"] in SERIALS_LIST:

        print(row["serial"])
        print(parts_df.loc[parts_df["serial"] == row["serial"]]["img_name"])
        scan = cv2.imread(os.path.join(images_folder, parts_df.loc[parts_df["serial"] == row["serial"]]["img_name"].item()), cv2.IMREAD_COLOR)
        processed_scan = cv2.imread(os.path.join(output_folder, row["serial"], "image1.tif"), cv2.IMREAD_COLOR)
        heatmap = cv2.cvtColor(cv2.imread(os.path.join(output_folder, row["serial"], "heatmap.tif"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # convert grayscale to green
        processed_scan[np.where(processed_scan[:,:,0] == 255)] = (0,255,0)

        # ax = plt.subplot(131)
        # ax.imshow()

        ax = plt.subplot(121)
        ax.set_title("Scan overlaid with segmented lamina in green")
        ax.imshow(cv2.addWeighted(scan, 0.9, processed_scan, 0.1, 0))

        ax = plt.subplot(122)
        ax.set_title(f"Heatmap: blue gained material, red lost. \n Changed px: {(100*row['prop_changed_px']):.2f}%")
        ax.imshow(heatmap)

        plt.suptitle(row["serial"])
        
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()