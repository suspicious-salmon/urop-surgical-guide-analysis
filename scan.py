import numpy as np
from matplotlib import pyplot as plt
import cv2
import imagej
from pathlib import Path
import os
# import jnius <- currently causes crash on import

import utility as u

print("Initialising Fiji...")
ij = imagej.init("sc.fiji:fiji", mode="interactive")
print("Done initialising Fiji")

def fiji2numpy(fiji_image, shape):
    numpy_image = np.array(
        fiji_image.getProcessor().getPixels(),
        dtype=np.uint8
    ).reshape(shape)
    return numpy_image

def to_imagej_path(path):
    return path.replace("\\", "/")

# currently doesn't work as jnius causes crash on import
# def numpy2fiji(numpy_image):
#     imagej_image = ij.py.to_java(numpy_image)
#     imagej_image = ij.dataset().create(imp)
#     imagej_image = ij.convert().convert(imp, ImagePlus)
#     return imagej_image

DEFAULT_DO_STEPS_DICT = {
    "subtract_background" : True,
    "first_close" : True,
    "max_val_scale" : True,
    "enhance_contrast" : True,
    "second_close" : True,
    "threshold" : True,
}

DEFAULT_STEPS_PARAMETERS_DICT = {
    "subtract_background" : 50,
    "first_close" : 11,
    "max_val_scale" : 122,
    "enhance_contrast" : 30,
    "second_close" : 31,
    "threshold" : 70,
}

DEFAULT_ALIGN_ARGUMENT = "run('Linear Stack Alignment with SIFT', 'initial_gaussian_blur=5 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=5 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate')"

def process_scan(in_directory,
                 out_directory,
                 do_steps_dict,
                 steps_parameters_dict,
                 save_steps=True):
    
    if save_steps:
        Path(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps")).mkdir(parents=True, exist_ok=True)

    imp = ij.IJ.openImage(in_directory)
    width, height = imp.shape[:2]

    # -------------
    # Subtract Background

    if do_steps_dict["subtract_background"]:
        ij.IJ.run(imp, "Subtract Background...", f"rolling={steps_parameters_dict['subtract_background']}")

    # -------------
        
    img = fiji2numpy(imp, shape=(height,width))

    # -------------
    # First Close

    if do_steps_dict["first_close"]:
        kernel_size = steps_parameters_dict["first_close"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        if save_steps:
            u.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im1.tif"), img)
    
    # ---------------
    # Max Val Scale
    
    if do_steps_dict["max_val_scale"]:
        max_val = steps_parameters_dict["max_val_scale"]
        img[img > max_val] = max_val
        img = (img.astype(float)*255/max_val).astype(np.uint8)

        if save_steps:
            u.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im3.tif"), img)

    # --------------

    # convert from numpy to imj image
    u.writeim(out_directory, img)
    imp = ij.IJ.openImage(out_directory)
    width, height = imp.shape[:2]

    # --------------
    # Enchance Contrast

    if do_steps_dict["enhance_contrast"]:
        ij.IJ.run(imp, "Enhance Contrast...", f"saturated={steps_parameters_dict['enhance_contrast']} normalize")

    # --------------

    img = fiji2numpy(imp, shape=(height,width))

    if save_steps:
        u.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im4.tif"), img)
    
    # --------------
    # Second Close

    if do_steps_dict["second_close"]:
        kernel_size = steps_parameters_dict["second_close"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        if save_steps:
            u.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im5.tif"), img)
        
    # --------------

    # --------------
    # Threshold

    if do_steps_dict["threshold"]:
        img = cv2.threshold(img, steps_parameters_dict["threshold"], 255, cv2.THRESH_BINARY)[1]

        if save_steps:
            u.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im6.tif"), img)
        
    # ---------------

    # output
    u.writeim(out_directory, img, overwrite=True)
    
    return img.shape[1], img.shape[0]
    
def align_sift(in_directory, out_directory, align_argument):
    macro = ";".join([
        f"File.openSequence('{to_imagej_path(in_directory)}')",
        align_argument,
        f"run('Image Sequence... ', 'dir=[{to_imagej_path(out_directory)}] format=TIFF')",
        "run('Close All')"
    ])

    ij.py.run_macro(macro)