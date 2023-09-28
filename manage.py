import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json

import _cad
import scan
import heatmap
import metrics
import _align
import _cvutil

output_folder = r"E:\greg\Results\Run8"
cad_folder = r"C:\temporary work folder gsk35\UIUCMxD\CAD"
images_folder = r"C:\temporary work folder gsk35\UIUCMxD\Images"

csv_directory = (r"C:\temporary work folder gsk35\UIUCMxD\mxd_258key.csv")
parts_df = pd.read_csv(csv_directory)

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

Path(output_folder).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "cads")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "aligned_scans")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "extra_pixels")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "missing_pixels")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "heatmaps")).mkdir(parents=True, exist_ok=True)
Path(os.path.join(output_folder, "steps")).mkdir(parents=True, exist_ok=True)

# write json file with details of operations and parameters
metadata_dict = {
    "do_steps_dict" : do_steps,
    "steps_parameters_dict" : steps_parameters,
}
if not os.path.isfile(os.path.join(output_folder, "metadata.json")):
    with open(os.path.join(output_folder, "metadata.json"), "x") as fp:
        json.dump(metadata_dict, fp)
else:
    print("WARNING: a metadata file already exists. Metadata will be saved in metadata_overwrite.json. It is at risk of being overwritten next time this code is run.")
    with open(os.path.join(output_folder, "metadata_overwrite.json"), "w") as fp:
        json.dump(metadata_dict, fp)

rows = parts_df.iterrows()
# rows = parts_df.loc[parts_df["serial"] == "GUIDE-0121-0143"].iterrows()
# rows = parts_df[parts_df.index.isin([235,236])].iterrows()

for idx, row in tqdm(rows, total=parts_df.shape[0]):
    try:
        width, height = _cvutil.get_file_image_dimensions(os.path.join(images_folder, row["img_name"]))

        # process and write cad file to destination folder. make it the same resolution as the corresponding scan file.
        img_cad = _cad.cad_to_img(os.path.join(cad_folder, row["mockingbird_file"]),
                                os.path.join(output_folder, "cads", row["serial"] + "_cad.tif"),
                                width, height)

        # align scan to cad
        _align.align_ccorr(
            os.path.join(output_folder, "cads", row["serial"] + "_cad.tif"),
            os.path.join(images_folder, row["img_name"]),
            {
                "aligned" : os.path.join(output_folder, "aligned_scans", row["serial"] + "_aligned.tif"),
                "extra_pixels" : os.path.join(output_folder, "extra_pixels", row["serial"] + "_extra.tif"),
                "missing_pixels" : os.path.join(output_folder, "missing_pixels", row["serial"] + "_missing.tif")
            }
        )

        # process and write scan file to destination folder
        imagejops.process_scan(os.path.join(output_folder, "aligned_scans", row["serial"] + "_aligned.tif"),
                               os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
                               do_steps,
                               steps_parameters,
                               save_steps=False)
        
        # processing introduces some artifacts around previous crop. remove these by cropping again with a slightly smaller kernel.
        _align.crop_to_inflated_cad(os.path.join(output_folder, "cads", row["serial"] + "_cad.tif"),
                                   os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
                                   os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
                                   kernel_size=275,
                                   overwrite=True)

        # heatmap
        heatmap.heatmap(os.path.join(output_folder, "cads", row["serial"] + "_cad.tif"),
                        os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
                        os.path.join(output_folder, "heatmaps", row["serial"] + "_heatmap.tif"))
        
    except OSError as e:
        print(f"Skipped {row['serial']} due to OSError. Error was: {e}")
        
metrics.get_directory_metrics(os.path.join(output_folder, "heatmaps"), output_folder)