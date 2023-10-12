"""This is the master file, """

import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import json

import _stl
import _scan
import _heatmap
import _metrics
import _align
import _cvutil

PX_PER_MM = 94.0 # the scale factor I used to scale up the surgical guide STL files.

def main(output_folder, cad_folder, images_folder, csv_directory):
    parts_df = pd.read_csv(csv_directory)

    # make folders to put everything in
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(output_folder, "cads")).mkdir(parents=False, exist_ok=True)
    Path(os.path.join(output_folder, "aligned_scans")).mkdir(parents=False, exist_ok=True)
    Path(os.path.join(output_folder, "extra_pixels")).mkdir(parents=False, exist_ok=True)
    Path(os.path.join(output_folder, "missing_pixels")).mkdir(parents=False, exist_ok=True)
    Path(os.path.join(output_folder, "heatmaps")).mkdir(parents=False, exist_ok=True)
    Path(os.path.join(output_folder, "steps")).mkdir(parents=False, exist_ok=True)

    # write json file with details of operations and parameters
    metadata_dict = {
        "do_steps_dict" : _scan.DEFAULT_DO_STEPS_DICT,
        "steps_parameters_dict" : _scan.DEFAULT_STEPS_PARAMETERS_DICT,
    }
    if not os.path.isfile(os.path.join(output_folder, "metadata.json")):
        with open(os.path.join(output_folder, "metadata.json"), "x") as fp:
            json.dump(metadata_dict, fp)
    else:
        print("WARNING: a metadata file already exists. Metadata will be saved in metadata_overwrite.json. It is at risk of being overwritten next time this code is run.")
        with open(os.path.join(output_folder, "metadata_overwrite.json"), "w") as fp:
            json.dump(metadata_dict, fp)

    rows = parts_df.iterrows()
    print("Making heatmaps")
    for idx, row in tqdm(rows, total=parts_df.shape[0]):
        try:
            width, height = _cvutil.get_file_image_dimensions(os.path.join(images_folder, row["img_name"]))

            # process and write cad file to destination folder. make it the same resolution as the corresponding scan file.
            _stl.cad_to_img(os.path.join(cad_folder, row["mockingbird_file"]),
                                    os.path.join(output_folder, "cads", row["serial"] + "_cad.tif"),
                                    width, height, px_per_mm=PX_PER_MM)

            # align scan to cad
            _align.align_ccorr(
                os.path.join(output_folder, "cads", row["serial"] + "_cad.tif"),
                os.path.join(images_folder, row["img_name"]),
                {
                    "aligned" : os.path.join(output_folder, "aligned_scans", row["serial"] + "_aligned.tif"),
                    "extra_pixels" : os.path.join(output_folder, "extra_pixels", row["serial"] + "_extra.tif"),
                    "missing_pixels" : os.path.join(output_folder, "missing_pixels", row["serial"] + "_missing.tif")
                },
                angle_bounds=(-60,0)
            )

            # segment aligned scan into binary image and write to destination folder
            _scan.process_scan(os.path.join(output_folder, "aligned_scans", row["serial"] + "_aligned.tif"),
                                os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
                                save_steps=False)
            
            # processing introduces some artifacts around previous crop. remove these by cropping again with a slightly smaller kernel.
            _align.crop_to_inflated_cad(os.path.join(output_folder, "cads", row["serial"] + "_cad.tif"),
                                    os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
                                    os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
                                    kernel_size=275,
                                    overwrite=True)

            # heatmap
            _heatmap.heatmap(os.path.join(output_folder, "cads", row["serial"] + "_cad.tif"),
                            os.path.join(output_folder, "steps", row["serial"] + "_processed.tif"),
                            os.path.join(output_folder, "heatmaps", row["serial"] + "_heatmap.tif"))
            
        except OSError as e:
            print(f"Skipped {row['serial']} due to OSError. Error was: {e}")
            
    print("Making metrics")
    _metrics.get_directory_metrics(os.path.join(output_folder, "heatmaps"), output_folder)

if __name__ == "__main__":
    root_dataset_folder = r"E:\greg\Dogs\UIUCMxD"
    root_output_folder = r"E:\greg\Organised Code\EnvTests\Results\Run1"
    
    cad_folder = os.path.join(root_dataset_folder, "CAD")
    images_folder = os.path.join(root_dataset_folder, "Images")
    csv_directory = os.path.join(root_dataset_folder, "mxd_258key.csv")

    main(root_output_folder, cad_folder, images_folder, csv_directory)
