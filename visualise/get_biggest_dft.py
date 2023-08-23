# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

import utility as u

%matplotlib qt

parts_csv_dir = r"C:\temporary work folder gsk35\UIUCMxD\mxd_258key.csv"
parts_df = pd.read_csv(parts_csv_dir)

column_sort = {
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

# %%
n_top = 3

for col_key in column_sort:
    column_sort[col_key] = parts_df.sort_values(col_key, key=abs, ascending=False, ignore_index=True)[:n_top]

# %%

for c in column_sort:
    print(f"Printing column {c}: \n{column_sort[c][c]} \n")
# %%

output_dir = r"C:\temporary work folder gsk35\Production\Run1"
heat_df = pd.read_json(os.path.join(output_dir, "metrics.json"))

# %%

for count, row in column_sort["DFT_slot2_thick_mm"].iterrows():
    ax = plt.subplot(221 + count)

    # ax.set_xlim((2600,3050))
    # ax.set_ylim((3100,2750)) # inverted limits to prevent image from being flipped
    ax.set_title(f"serial : {row['serial']} \n"
                f"DFT_slot2_thick_mm : {row['DFT_slot2_thick_mm']:.4f} \n")
    
    img = u.readim(os.path.join(output_dir, row["serial"], "heatmap.tif"))
    ax.imshow(img)

plt.tight_layout()
plt.show()
# %%
