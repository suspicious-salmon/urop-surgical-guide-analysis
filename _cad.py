import trimesh
from trimesh import viewer
import cv2
import matplotlib.pyplot as plt
import math
import os
import numpy as np

import _cvutil

def pad_image(image, target_width, target_height, pad_value=0):
    """Pads grayscale image with equal amount of pixels on top & bottom and on left & right.

    Args:
        image (numpy array): grayscale opencv image to be padded
        target_width (int): target width
        target_height (int): target height
        pad_value (int, optional): intensity value (0-255) to fill padded area with. Defaults to 0.

    Returns:
        numpy array: padded opencv image
    """
    pad_vertical = target_height-image.shape[0]
    pad_horizontal = target_width-image.shape[1]
    image = cv2.copyMakeBorder(image,
                               math.floor(pad_vertical/2),
                               math.ceil(pad_vertical/2),
                               math.floor(pad_horizontal/2),
                               math.ceil(pad_horizontal/2),
                               cv2.BORDER_CONSTANT, value=pad_value)
    return image

def resize_image(image, target_dimensions, pad_value=0):
    """Resizes image while maintaining aspect ratio. To match target dimensions, any remaining space after scaling is filled with padding.

    Args:
        image (numpy array): opencv image to resize
        target_dimensions (tuple): (width, height) of target image

    Returns:
        numpy array: padded opencv image
    """
    final_width, final_height = target_dimensions
    width, height = image.shape[1], image.shape[0]
    # find dimensions to scale image by without changing aspect ratio or making it too big
    if final_width/width > final_height/height:
        scale_dimensions = (int(width * final_height/height), int(final_height))
    else:
        scale_dimensions = (int(final_width), int(height * final_width/width))
        
    image = cv2.resize(image, scale_dimensions, interpolation=cv2.INTER_NEAREST)
    image = pad_image(image, final_width, final_height, pad_value)
    return image

PX_PER_MM = 94.0 # the scale factor I used to scale up the surgical guide STL files.
VIEW_SCALE = 5 # for ease of viewing in matplotlig, useful while debugging
d = 0 # extra space on each side, in mm
def save_path2D(slice_2D, img_directory, px_per_mm = PX_PER_MM):
    """Inspired by trimesh source code, outputs the path2D at the correct pixels per inch as a black-and-white image.
    note, this function is slightly flawed. When plotted in matplotlib, edges must be thick enough to fill the part with white later.
    This results in a slightly bigger outline than in reality of appxoimately <> pixels, or <> mm.

    Args:
        slice_2D (trimesh.path.Path2D): 2D path slice to save
        img_directory (directory string): directory to save slice image to
        px_per_mm (_type_, optional): Scale factor of slice; each mm will be saved as px_per_mm pixels. Defaults to PX_PER_MM.

    Raises:
        OSError: if a file at img_directory already exists
    """
    
    min_x = slice_2D.vertices[:,0].min() - d
    max_x = slice_2D.vertices[:,0].max() + d
    min_y = slice_2D.vertices[:,1].min() - d
    max_y = slice_2D.vertices[:,1].max() + d
    width = max_x - min_x
    aspect_ratio = width / (max_y - min_y)
    dpi = (width * px_per_mm) / (VIEW_SCALE * aspect_ratio)

    fig = plt.figure(frameon = False)
    fig.set_size_inches(VIEW_SCALE*aspect_ratio, VIEW_SCALE)

    # make figure without axes
    axis = plt.Axes(fig, [0.,0.,1.,1.])
    axis.set_axis_off()
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    fig.add_axes(axis)

    # plot path2D
    for points in slice_2D.discrete:
        axis.plot(*points.T, color="k", linewidth=0.2)

    # save as image
    if not os.path.isfile(img_directory):
        fig.savefig(img_directory, dpi=dpi)
    else:
        raise OSError("Cad outline image already exists")

def cad_to_img(stl_directory, img_directory, target_width=None, target_height=None, px_per_mm=None):
    """Converts STL file from stl_directory to a black-and-white image at dimensions matching EITHER:
    - the specified target_width and target_height
    - the specified scale, px_per_mm
    and saves it at img_directory.

    Args:
        stl_directory (directory string): directory to read STL file from
        img_directory (directory string): directory to save resulting image to
        target_width (int, optional): target width of output image. Defaults to None.
        target_height (int, optional): target height of output image. Defaults to None.
        px_per_mm (float, optional): if target_width and target_height are None, scale factor to use from stl to output image. Defaults to None.
    """

    # load stl file
    mesh = trimesh.load_mesh(stl_directory)

    mesh.visual.face_colors = [0,0,0,255]
    # move mesh to centre of coordinate system
    mesh.vertices -= mesh.center_mass

    # slice stl and rotate cross-section
    myslice = mesh.section(plane_origin=[0,0,0],
                        plane_normal=[0,1,0])
    # the next 2 lines are hard-coded for the surgical guide STL files. They rotate the STL file so its cross-section faces the camera.
    myslice.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(270), [1,0,0], point=myslice.centroid))
    myslice.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(180), [0,1,0], point=myslice.centroid))
    section, _ = myslice.to_planar()

    # save stl as image, then read again (couldn't find a nice way to export straight from matplotlib to opencv). weirdly, all the information of the image gets saved by matplotlib in its alpha channel. So I take only this channel for the grayscale one.
    save_path2D(section, img_directory[:-4]+"_mpl.tif", px_per_mm=px_per_mm)
    img = _cvutil.readim(img_directory[:-4]+"_mpl.tif", cv2.IMREAD_UNCHANGED)[:,:,3]

    # for some reason despite img having the same dtype and shape as a grayscale image, opencv floodfill does nothing unless this is done.
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # flood fill to make the part all white and invert (fills at coordinate 200px from bottom-left corner)
    thresh_lo = 200
    thresh_hi = 200
    cv2.floodFill(img, None, (200, img.shape[0]-200), (255,255,255,0), loDiff=(thresh_lo, thresh_lo, thresh_lo), upDiff=(thresh_hi, thresh_hi, thresh_hi), flags=cv2.FLOODFILL_FIXED_RANGE)

    # pad to make right size
    if target_height is not None and target_width is not None:
        img = pad_image(img, target_width, target_height)

    # output
    _cvutil.writeim(img_directory, img, overwrite=True)