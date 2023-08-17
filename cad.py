# can't load qt so doesn't work in ubuntu

import trimesh
from trimesh import viewer
import numpy as np   
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math

# SCAN_WIDTH, SCAN_HEIGHT = 4901, 4779
# SCAN_WIDTH, SCAN_HEIGHT = 5313, 5158

def pad_image(image, target_width, target_height, pad_value=0):
    pad_vertical = target_height-image.shape[0]
    pad_horizontal = target_width-image.shape[1]
    image = cv2.copyMakeBorder(image,
                               math.floor(pad_vertical/2),
                               math.ceil(pad_vertical/2),
                               math.floor(pad_horizontal/2),
                               math.ceil(pad_horizontal/2),
                               cv2.BORDER_CONSTANT, value=pad_value)
    return image

def resize_image(image, target_dimensions):
    final_width, final_height = target_dimensions
    width, height = image.shape[1], image.shape[0]
    # find dimensions to scale image by without changing aspect ratio or making it too big
    if final_width/width > final_height/height:
        scale_dimensions = (int(width * final_height/height), int(final_height))
    else:
        scale_dimensions = (int(final_width), int(height * final_width/width))
        
    image = cv2.resize(image, scale_dimensions, interpolation=cv2.INTER_LINEAR)
    image = pad_image(image, final_width, final_height, 0)
    return image

# inspired by trimesh source code, outputs the path at the correct pixels per inch
PX_PER_MM = 94.0
VIEW_SCALE = 5
def save_path2D(slice_2D, img_directory):
    min_x = slice_2D.vertices[:,0].min()
    max_x = slice_2D.vertices[:,0].max()
    min_y = slice_2D.vertices[:,1].min()
    max_y = slice_2D.vertices[:,1].max()
    width = max_x - min_x
    aspect_ratio = width / (max_y - min_y)
    dpi = (width * PX_PER_MM) / (VIEW_SCALE * aspect_ratio)

    fig = plt.figure(frameon = False)
    fig.set_size_inches(VIEW_SCALE*aspect_ratio, VIEW_SCALE)

    axis = plt.Axes(fig, [0.,0.,1.,1.])
    axis.set_axis_off()
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(min_y, max_y)
    fig.add_axes(axis)

    for points in slice_2D.discrete:
        axis.plot(*points.T, color="k", linewidth=0.2)

    fig.savefig(img_directory, dpi=dpi)

def cad_to_img(stl_directory, img_directory, scan_width, scan_height):

    mesh = trimesh.load_mesh(stl_directory)

    mesh.visual.face_colors = [0,0,0,255]
    # move mesh to centre of coordinate system
    mesh.vertices -= mesh.center_mass

    # slice stl and rotate cross-section
    myslice = mesh.section(plane_origin=[0,0,0],
                        plane_normal=[0,1,0])
    myslice.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(270), [1,0,0], point=myslice.centroid))
    myslice.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(180), [0,1,0], point=myslice.centroid))
    section, _ = myslice.to_planar()

    # save stl as image, then read again (couldn't find a nice way to export straight from matplotlib to opencv). weirdly, all the information of the image gets saved by matplotlib in its alpha channel. So I take only this channel for the grayscale one.
    save_path2D(section, img_directory)
    img = cv2.imread(img_directory, cv2.IMREAD_UNCHANGED)[:,:,3]

    # for some reason despite img having the same dtype and shape as a grayscale image, opencv floodfill does nothing unless this is done.
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # flood fill to make the part all white and invert (fills at coordinate 200px from bottom-left corner)
    thresh_lo = 200
    thresh_hi = 200
    cv2.floodFill(img, None, (200, img.shape[0]-200), (255,255,255,0), loDiff=(thresh_lo, thresh_lo, thresh_lo), upDiff=(thresh_hi, thresh_hi, thresh_hi), flags=cv2.FLOODFILL_FIXED_RANGE)

    # pad to make right size
    img = pad_image(img, scan_width, scan_height)

    # output
    cv2.imwrite(img_directory, img)