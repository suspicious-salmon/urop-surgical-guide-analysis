# %%

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd

%matplotlib qt

output_dir = r"C:\temporary work folder gsk35\Production\Run1"
heat_df = pd.read_json(os.path.join(output_dir, "metrics.json"))

#%%

print(heat_df)
# %%

sorted_lost_df = heat_df.sort_values("prop_lost_px", ascending=False, ignore_index=True)
print(sorted_lost_df)
# %%

sorted_gained_df = heat_df.sort_values("prop_gained_px", ascending=False, ignore_index=True)
sorted_total_df = heat_df.sort_values("prop_changed_px", ascending=False, ignore_index=True)

# %%
def display_df(df):
    for counter, row in df.iterrows():
        img = cv2.imread(os.path.join(output_dir, row['serial'], "heatmap.tif"), cv2.IMREAD_COLOR)
        
        ax = plt.subplot(221 + counter)
        # ax.set_xlim((2600,3050))
        # ax.set_ylim((3100,2750)) # inverted limits to prevent image from being flipped
        ax.set_title(f"serial : {row['serial']} \n"
                    f"prop_lost_px : {row['prop_lost_px']:.4f} \n"
                    f"prop_gained_px : {row['prop_gained_px']:.4f} \n"
                    f"prop_changed_px : {row['prop_changed_px']:.4f}")
        ax.imshow(img)

        counter += 1
    plt.tight_layout()
    plt.show()
# %%

display_df(sorted_gained_df[:4])
# %%
