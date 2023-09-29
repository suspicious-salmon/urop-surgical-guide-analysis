"""This contains the pipeline I used to segment the surgical guide photos. I constantly save and read images in the same function, as an alternative to converting from a numpy array to a pyimagej image (which I could not get working)."""

import numpy as np
import cv2
import imagej
from pathlib import Path
import os
# import jnius <- currently causes crash on import

import _cvutil

# Initialise fiji. I had to use interactive mode to make it work - imagej windows will open in the windows UI while pyimagej functions are running.
print("Initialising Fiji...")
ij = imagej.init("sc.fiji:fiji", mode="interactive")
print("Done initialising Fiji")

def fiji2numpy(fiji_image, shape):
    """Convert pyimagej image to numpy array

    Args:
        fiji_image: image loaded using ij.ij.openImage()
        shape (tuple): (height, width) of image"""
    
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
    "first_close" : False,
    "max_val_scale" : True,
    "enhance_contrast" : False,
    "second_close" : True,
    "threshold" : True,
}

DEFAULT_STEPS_PARAMETERS_DICT = {
    "subtract_background" : 50, # rolling ball radius
    "first_close" : None, # kernel size of first morphological close
    "max_val_scale" : 122, # intensity to truncate photo at (i.e. all intensities above max_val_scale will be set to max_val_scale)
    "enhance_contrast" : None, # saturated pixels percentage, same as Imagej parameter in enhance contrast tol
    "second_close" : 31, # kernel size of second morphological close
    "threshold" : 40, # threshold value for binary threshold. any pixels below threshold are set to zero, any above to 255.
}

DEFAULT_ALIGN_ARGUMENT = "run('Linear Stack Alignment with SIFT', 'initial_gaussian_blur=5 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=5 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate')"

def process_scan(in_directory,
                 out_directory,
                 do_steps_dict=DEFAULT_DO_STEPS_DICT,
                 steps_parameters_dict=DEFAULT_STEPS_PARAMETERS_DICT,
                 save_steps=True):
    """Apply the image segmentation pipeline to the photo of the surgical device.

    Args:
        in_directory (str): directory of surgical device photo to read
        out_directory (str): directory of segmented result to save to
        do_steps_dict (dict, optional): dictionary specifying which steps of the pipeline to include. Defaults to DEFAULT_DO_STEPS_DICT, which is what I used in the final pipeline.
        steps_parameters_dict (dict, optional)): dictionary specifying the parameter for each step of the pipeline. Defaults to DEFAULT_DO_STEPS_DICT, which is what I used in the final pipeline.
        save_steps (bool, optional): If true, will save intermediate steps in the segmentation process in their own folder. Useful for debugging. Defaults to True.

    Returns:
        tuple: dimensions (width, height) of segmented result that was saved to out_directory
    """
    
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
            _cvutil.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im1.tif"), img)
    
    # ---------------
    # Max Val Scale
    
    if do_steps_dict["max_val_scale"]:
        max_val = steps_parameters_dict["max_val_scale"]
        img[img > max_val] = max_val
        img = (img.astype(float)*255/max_val).astype(np.uint8)

        if save_steps:
            _cvutil.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im3.tif"), img)

    # --------------

    # convert from numpy to imj image
    _cvutil.writeim(out_directory, img)
    imp = ij.IJ.openImage(out_directory)
    width, height = imp.shape[:2]

    # --------------
    # Enchance Contrast

    if do_steps_dict["enhance_contrast"]:
        ij.IJ.run(imp, "Enhance Contrast...", f"saturated={steps_parameters_dict['enhance_contrast']} normalize")

    # --------------

    img = fiji2numpy(imp, shape=(height,width))

    if save_steps:
        _cvutil.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im4.tif"), img)
    
    # --------------
    # Second Close

    if do_steps_dict["second_close"]:
        kernel_size = steps_parameters_dict["second_close"]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        if save_steps:
            _cvutil.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im5.tif"), img)
        
    # --------------

    # --------------
    # Threshold

    if do_steps_dict["threshold"]:
        img = cv2.threshold(img, steps_parameters_dict["threshold"], 255, cv2.THRESH_BINARY)[1]

        if save_steps:
            _cvutil.writeim(os.path.join(("\\".join(out_directory.split("\\")[:-1])), "steps\\im6.tif"), img)
        
    # ---------------

    # output
    _cvutil.writeim(out_directory, img, overwrite=True)
    
    return img.shape[1], img.shape[0]
    
# note, I do not use SIFT alignment anymore, this is a redundant function.
def align_sift(in_directory, out_directory, align_argument):
    macro = ";".join([
        f"File.openSequence('{to_imagej_path(in_directory)}')",
        align_argument,
        f"run('Image Sequence... ', 'dir=[{to_imagej_path(out_directory)}] format=TIFF')",
        "run('Close All')"
    ])

    ij.py.run_macro(macro)