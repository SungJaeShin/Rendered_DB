# ===========================================================
# Define feature extraction and descriptor
# Following best method 
#       => https://github.com/SungJaeShin/Feature_matching
# ===========================================================

import torch
import cv2 as cv
import numpy as np
from typing import Union
import PIL
from PIL import Image
from skimage.feature import daisy
from models.RoMa import roma_model

import pdb

def cvt_rgb_to_gray(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray

# Default extract method is ORB
def feature_extractor(method, img):
    # [ORB Feature Method]
    orb = cv.ORB_create()
    img_kpt = orb.detect(img)

    # [SIFT Feature Method]
    if method == 1:
        sift = cv.SIFT_create()
        img_kpt = sift.detect(img, None)
    # [AKAZE Feature Method]
    if method == 2:
        akaze = cv.AKAZE_create()
        img_kpt = akaze.detect(img)

    return img_kpt

# For using DAISY descriptors 
def compute_daisy_descriptors(image, keypoints):
    descriptors = daisy(image, step=1, radius=15, rings=3, histograms=8, orientations=8, visualize=False)
    keypoints = np.array([kp.pt for kp in keypoints], dtype=np.int32)
    key_descriptors = [descriptors[pt[1], pt[0]] for pt in keypoints if 0 <= pt[0] < image.shape[1] and 0 <= pt[1] < image.shape[0]]
    return np.array(key_descriptors, dtype=np.float32)

def descriptor_extractor(method, img, kpt):
    # [ORB Descriptor Method]
    orb = cv.ORB_create()
    _, img_des = orb.detectAndCompute(img, None)
    
    # [Patented SURF Method] (Not Use)
    if method == 1:
        surf = cv.xfeatures2d.SURF_create()
        _, img_des = surf.compute(img, kpt)
    # [DAISY Descriptor Method]
    if method == 2:
        img_des = compute_daisy_descriptors(img, kpt)
    # [AKAZE Descriptor Method]
    if method == 3:
        akaze = cv.AKAZE_create()
        _, img_des = akaze.compute(img, kpt)

    return img_des

def roma_based_extractor(img1, img2, config, device):
    # Get Pre-trained Model (roma & dino)
    roma_weights = torch.load(config['learning_method']['indoor'], map_location=device)
    dinov2_weights = torch.load(config['learning_method']['dinov2'], map_location=device)
    
    # Set basic model parameter
    coarse_res = (img1.shape[0],img1.shape[1])
    upsample_res = (864, 864)

    # Get Pre-trained RoMa Model
    roma_model = roma_model(resolution=coarse_res, upsample_preds=True, weights=roma_weights,
                            dinov2_weights=dinov2_weights, device=device, amp_dtype=torch.float16)
    roma_model.upsample_res = upsample_res

    # Get Output Resolution
    H, W = roma_model.get_output_resolution() 

    # Change image size to fit output resolution
    im1 = img1.resize((W, H))
    im2 = img2.resize((W, H))

    # Get warp and convariance (certainty)
    # ============================================ 
    # Example resolution : 560 
    #   => (1) warp size: torch.Size([560, 1120, 4]) & type: <class 'torch.Tensor'>
    #   => (2) certainty size: torch.Size([10000]) & type: <class 'torch.Tensor'>
    warp, certainty = roma_model.match(im1, im2, device=device)
        
    # Match Two Images
    # matches size: torch.Size([10000, 4]) & type: <class 'torch.Tensor'>
    matches, certainty = roma_model.sample(warp, certainty)

    # Get Feature Points
    # ============================================ 
    # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
    #   => kptsA size: torch.Size([10000, 2]) & type: <class 'torch.Tensor'>
    #   => kptsB size: torch.Size([10000, 2]) & type: <class 'torch.Tensor'>
    kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H, W, H, W)

    return kptsA, kptsB, warp, matches


        
