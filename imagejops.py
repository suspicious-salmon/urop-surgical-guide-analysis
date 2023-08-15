import numpy as np
from matplotlib import pyplot as plt
import cv2
import imagej
from pathlib import Path

import os
# import scyjava as sj
# import jnius <- currently causes crash on import

print("Initialising Fiji...")
ij = imagej.init("sc.fiji:fiji", mode="interactive")
# FolderOpener = sj.jimport('ij.plugin.FolderOpener')
# StackWriter = sj.jimport('ij.plugin.StackWriter')
# ImagePlus = jnius.autoclass('ij.ImagePlus')
print("Done initialising Fiji")

def fiji2numpy(fiji_image, shape):
    numpy_image = np.array(
        fiji_image.getProcessor().getPixels(),
        dtype=np.uint8
    ).reshape(shape)
    return numpy_image

def to_imagej_path(path):
    return path.replace("\\", "/")

# currently cannot be used as jnius causes crash on import
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

def process_scan(in_directory,
                 out_directory,
                 do_steps_dict,
                 steps_parameters_dict,
                 save_steps=True):
    
    if save_steps:
        Path(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps")).mkdir(parents=True, exist_ok=True)
    
    # if do_steps_dict is None:
    #     do_steps_dict = DEFAULT_DO_STEPS_DICT
    # if steps_parameters_dict is None:
    #     steps_parameters_dict = DEFAULT_STEPS_PARAMETERS_DICT

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
            if not cv2.imwrite(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im1.tif"), img):
                raise Exception("Couldn't save im1")
    
    # -------------

    # bilateral blur
    # BILATERAL_BLUR_SIZE = 11
    # img = cv2.bilateralFilter(img, BILATERAL_BLUR_SIZE, BILATERAL_BLUR_SIZE, BILATERAL_BLUR_SIZE)

    # if not cv2.imwrite(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "im2.tif"), img):
    #     raise Exception("Couldn't save im2")
    
    # ---------------
    # Max Val Scale
    
    if do_steps_dict["max_val_scale"]:
        max_val = steps_parameters_dict["max_val_scale"]
        img[img > max_val] = max_val
        img = (img.astype(float)*255/max_val).astype(np.uint8)

        if save_steps:
            if not cv2.imwrite(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im3.tif"), img):
                raise Exception("Couldn't save im3")

    # --------------

    # convert from numpy to imj image
    if not cv2.imwrite(out_directory, img):
        raise Exception(f"Step 1and2 image from {in_directory} could not be written to {out_directory}")
    imp = ij.IJ.openImage(out_directory)
    width, height = imp.shape[:2]

    # --------------
    # Enchance Contrast

    if do_steps_dict["enhance_contrast"]:
        ij.IJ.run(imp, "Enhance Contrast...", f"saturated={steps_parameters_dict['enhance_contrast']} normalize")

    # --------------

    img = fiji2numpy(imp, shape=(height,width))

    if save_steps:
        if not cv2.imwrite(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im4.tif"), img):
            raise Exception("Couldn't save im4")
    
    # --------------
    # Second Close

    if do_steps_dict["second_close"]:
        kernel_size = steps_parameters_dict["second_close"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        if save_steps:
            if not cv2.imwrite(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im5.tif"), img):
                raise Exception("Couldn't save im5")
        
    # --------------

    # --------------
    # Threshold

    if do_steps_dict["threshold"]:
        img = cv2.threshold(img, steps_parameters_dict["threshold"], 255, cv2.THRESH_BINARY)[1]

        if save_steps:
            if not cv2.imwrite(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im6.tif"), img):
                raise Exception("Couldn't save im6")
        
    # ---------------

    # output
    if not cv2.imwrite(out_directory, img):
        raise Exception(f"Step 3and4 image from {out_directory} could not be written to {out_directory}")
    
    return img.shape[1], img.shape[0]
    
def align(in_directory, out_directory):
    macro = ";".join([
        f"File.openSequence('{to_imagej_path(in_directory)}')",
        # "run('Linear Stack Alignment with SIFT', 'initial_gaussian_blur=5 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=5 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate')",
        "run('Linear Stack Alignment with SIFT', 'initial_gaussian_blur=3 steps_per_scale_octave=10 minimum_image_size=32 maximum_image_size=128 feature_descriptor_size=5 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=10 inlier_ratio=0.05 expected_transformation=Rigid interpolate')",
        f"run('Image Sequence... ', 'dir=[{to_imagej_path(out_directory)}] format=TIFF')",
        "run('Close All')"
    ])

    ij.py.run_macro(macro)