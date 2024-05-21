from PIL import Image
import numpy as np
import PIL
import torch
import cv2 as cv

import pdb

def draw_keypoints_on_image(image, keypoints, color='blue', radius=2, use_normalized_coordinates=False):
    """
    Draws keypoints on an image.
        
    Args:
        image: a PIL.Image object.
        keypoints: a numpy array with shape [num_keypoints, 2].
        color: color to draw the keypoints with. Default is red.
        radius: keypoint radius. Default value is 2.
        use_normalized_coordinates: if True (default), treat keypoint values as
        relative to the image.  Otherwise treat them as absolute.
    """
    draw = PIL.ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = tuple([im_width * x for x in keypoints_x])
        keypoints_y = tuple([im_height * y for y in keypoints_y])
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                    (keypoint_x + radius, keypoint_y + radius)],
                    outline=color, fill=color)

def draw_correspondences(image, kptsA, kptsB, color='red', size=0):
    """
    Draws keypoints on an image.
        
    Args:
        image: a PIL.Image object.
        keypoints: a numpy array with shape [num_keypoints, 2].
        color: color to draw the keypoints with. Default is red.
        size: correspondence line size between two keypoints 
    """
    draw = PIL.ImageDraw.Draw(image)
    length = kptsA.shape[0]
    for i in range(length):
        correspondence = [(kptsA[i][0], kptsA[i][1]), (kptsB[i][0], kptsB[i][1])]
        draw.line(correspondence, fill=color, width=size)
