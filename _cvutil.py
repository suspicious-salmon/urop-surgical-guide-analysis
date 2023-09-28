"""This module contains functions that augment opencv's imread() and imwrite() functions.
Firstly, it raises and error if the read or write was unsuccessful, making debugging easier.
Secondly, it prevents accidental overwriting by default, raising an error if the file to write already exists.
Thirdly, it allows for reading images whose filenames contain non-ascii characters, which would normally be unsuccessful in vanilla opencv."""

import cv2
import os
import numpy as np

def readim(dir, readtype):
    """Reads image just like opencv's imread() but raises error if read was unsuccessful. Also can read images with non-ascii filenames which opencv cannot ordinarily do.

    Args:
        dir (directory string): directory to read image from
        readtype (cv::ImreadModes): opencv read type, e.g. cv2.IMREAD_UNCHANGED

    Raises:
        OSError: If image could not be read.

    Returns:
        numpy array: image
    """

    # opencv doesn't support unicode in filenames. if image contains characters not in ASCII, must use something other than opencv to read the file in
    try:
        bytes(dir, "ascii")
    except UnicodeEncodeError:
        img = cv2.imdecode(np.fromfile(dir, dtype=np.uint8), readtype)
    else:
        img = cv2.imread(dir, readtype)

    if img is None:
        raise OSError(f"Could not read image at {dir}")
    else:
        return img

def writeim(dir, img, overwrite=False):
    """Writes image just like opencv's imwrite() but raises error if write was unsuccessful, or if the file already exists.

    Args:
        dir (directory string): directory to write image to
        img (numpy array): opencv image
        overwrite (bool, optional): If true, writeim() will raise OSError if file to write already exists. Defaults to False.

    Raises:
        OSError: if write failed or, if overwrite=False, if file already exists
    """
    if not overwrite and os.path.isfile(dir):
        raise OSError(f"Could not write image to {dir}, image alreay exists!")
    else:
        if not cv2.imwrite(dir, img):
            raise OSError(f"Could not write image to {dir}")   
    
def get_file_image_dimensions(img_dir):
    """Reads the image at directory img_dir and returns its dimensions.

    Args:
        img_dir (directory string): directory to read image from

    Returns:
        tuple: (width, height)
    """

    img = readim(img_dir, cv2.IMREAD_GRAYSCALE)
    return img.shape[1], img.shape[0]