import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json

import cad
import imagejops
import heatmap
import metrics

output_folder = r"E:\greg\Run4"
cad_folder = r"C:\temporary work folder gsk35\UIUCMxD\CAD"
images_folder = r"C:\temporary work folder gsk35\UIUCMxD\Images"

Path(output_folder).mkdir(parents=True, exist_ok=True) # exist_ok=False to prevent overwriting any directories by accident

csv_directory = (r"C:\temporary work folder gsk35\UIUCMxD\mxd_258key.csv")
parts_df = pd.read_csv(csv_directory)

# choose which operations will be done and which parameters they will use
# do_steps_dict = imagejops.DEFAULT_DO_STEPS_DICT
# steps_parameters_dict = imagejops.DEFAULT_STEPS_PARAMETERS_DICT

do_steps = {
    "subtract_background" : True,
    "first_close" : False,
    "max_val_scale" : True,
    "enhance_contrast" : False,
    "second_close" : True,
    "threshold" : True,
}

steps_parameters = {
    "subtract_background" : 50,
    "first_close" : None,
    "max_val_scale" : 122,
    "enhance_contrast" : None,
    "second_close" : 31,
    "threshold" : 40,
}

sift_align_argument = imagejops.DEFAULT_ALIGN_ARGUMENT

# write json file with details of operations and parameters
metadata_dict = {
    "do_steps_dict" : do_steps,
    "steps_parameters_dict" : steps_parameters,
    "sift_align_argument" : sift_align_argument
}
with open(os.path.join(output_folder, "metadata.json"), "x") as fp:
    json.dump(metadata_dict, fp)

rows = parts_df.iterrows()
# rows = parts_df.loc[parts_df["serial"] == "GUIDE-0121-0143"].iterrows()

for idx, row in tqdm(rows, total=parts_df.shape[0]):
    # make destination folder for the two files
    dest_folder_path = os.path.join(output_folder, row["serial"])
    Path(dest_folder_path).mkdir(parents=True, exist_ok=False)

    # process and write scan file to destination folder
    width, height = imagejops.process_scan(os.path.join(images_folder, row["img_name"]),
                                           os.path.join(dest_folder_path, "image1.tif"),
                                           do_steps,
                                           steps_parameters)

    # process and write cad file to destination folder
    img_cad = cad.cad_to_img(os.path.join(cad_folder, row["mockingbird_file"]),
                             os.path.join(dest_folder_path, "image0.tif"),
                             width, height)
    
    # align cad & scan
    Path(os.path.join(dest_folder_path, "aligned")).mkdir(parents=True, exist_ok=False)
    imagejops.align(dest_folder_path, os.path.join(dest_folder_path, "aligned"), sift_align_argument)

    # heatmap
    heatmap.heatmap(os.path.join(dest_folder_path, "image0.tif"),
                    os.path.join(dest_folder_path, "aligned", "Aligned 2 of 20001.tif"),
                    os.path.join(dest_folder_path, "heatmap.tif"))
    
metrics.get_directory_metrics(output_folder)