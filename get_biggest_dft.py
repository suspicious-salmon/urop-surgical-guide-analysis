"""This script sorts the parts, for each feature measurement that Bill King took,
in order of descending DFT (difference from target) magnitude.
It then allows you to view the corresponding heatmaps.
Useful to try to find parts that might have large errors."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

import _cvutil

parts_csv_dir = r"C:\temporary work folder gsk35\UIUCMxD\mxd_258key.csv"
parts_df = pd.read_csv(parts_csv_dir)
parts_results_dir = r"C:\temporary work folder gsk35\Production\Run8"
heatmap_df = pd.read_json(os.path.join(parts_results_dir, "metrics.json"))

# each entry will contain parts_df sorted by descending magnitude of the entry's corresponding column
DFTs = {
    "DFT_theta" : None,
    "DFT_t" : None,
    "DFT_l" : None,
    "DFT_d" : None,
    "DFT_slot_width_mm" : None,
    "DFT_slot_thick_mm" : None,
    "DFT_slot2_thick_mm" : None,
    "DFT_slot2_depth_mm" : None,
    "DFT_slot_slot_dist_mm" : None,
    "DFT_hole0_diam_mm" : None,
    "DFT_hole1_diam_mm" : None,
    "DFT_hole_hole_dist_mm" : None,
    "DFT_hole0_slot_dist_mm" : None,
    "DFT_hole1_slot_dist_mm" : None,
    "DFT_top_flange_thick_mm" : None,
    "DFT_slot_slot_angle_deg" : None,
}

n_top = 3
for col_key in DFTs:
    DFTs[col_key] = parts_df.sort_values(col_key, key=abs, ascending=False, ignore_index=True)
    print(f"Biggest {n_top} DFTs in column {col_key}: \n{DFTs[col_key][col_key][:n_top]} \n")

# pick a column (here DFT_slot2_thick_mm) and view heatmaps sorted by highest DFTs in bill king's excel file
for count, row in DFTs["DFT_slot2_thick_mm"].iterrows():
    plt.subplot(221 + count)
    plt.title(f"serial : {row['serial']} \nDFT_slot2_thick_mm : {row['DFT_slot2_thick_mm']:.4f} \n")
    
    img = _cvutil.readim(os.path.join(parts_results_dir, "heatmaps", f"{row['serial']}_heatmap.tif"), cv2.IMREAD_COLOR)
    plt.imshow(img)

plt.tight_layout()
plt.show()