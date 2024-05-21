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

from .visualize_utils import draw_keypoints_on_image, draw_correspondences

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

def roma_based_extractor(img1, img2, db, config, device):
    # Get Pre-trained Model (roma & dino)
    roma_weights = torch.load(config['learning_method']['outdoor'], map_location=device)
    dinov2_weights = torch.load(config['learning_method']['dinov2'], map_location=device)
    
    # Set basic model parameter
    coarse_res = (560, 560)
    upsample_res = (img1.shape[0], img1.shape[1])

    # Get Pre-trained RoMa Model
    model = roma_model(resolution=coarse_res, upsample_preds=True, weights=roma_weights,
                            dinov2_weights=dinov2_weights, device=device, amp_dtype=torch.float16)
    model.upsample_res = upsample_res

    # Get Output Resolution
    H, W = model.get_output_resolution() 

    # Change image size to fit output resolution
    im1 = Image.fromarray(img1).resize((W, H))
    im2 = Image.fromarray(img2).resize((W, H))

    # Get warp and convariance (certainty)
    # ============================================ 
    # Example resolution : 560 
    #   => (1) warp size: torch.Size([560, 1120, 4]) & type: <class 'torch.Tensor'>
    #   => (2) certainty size: torch.Size([10000]) & type: <class 'torch.Tensor'>
    warp, certainty = model.match(im1, im2, device=device)
        
    # Match Two Images
    # matches size: torch.Size([10000, 4]) & type: <class 'torch.Tensor'>
    matches, certainty = model.sample(warp, certainty)

    # Get Feature Points
    # ============================================ 
    # Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
    #   => kptsA size: torch.Size([10000, 2]) & type: <class 'torch.Tensor'>
    #   => kptsB size: torch.Size([10000, 2]) & type: <class 'torch.Tensor'>
    kptsA, kptsB = model.to_pixel_coordinates(matches, H, W, H, W)

    # Find a fundamental matrix (or anything else of interest) to get Inliner Feature Points
    # ============================================ 
    F, mask = cv.findFundamentalMat(kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.01, 
                                     method=cv.USAC_MAGSAC, confidence=0.999999, maxIters=10000)

    # Select inlier points
    # ============================================ 
    kptsA = kptsA[mask.ravel()==1]
    kptsB = kptsB[mask.ravel()==1]
    
    # Eamples 
    # ============================================ 
    # use numpy to convert the pil_image into a numpy array
    numpy_image1 = np.array(im1)  
    numpy_image2 = np.array(im2)  
        
    # convert to a openCV2 image and convert from RGB to BGR format
    cv_image1 = cv.cvtColor(numpy_image1, cv.COLOR_RGB2BGR)
    cv_image2 = cv.cvtColor(numpy_image2, cv.COLOR_RGB2BGR)
        
    # ================= Move feature points for plotting correspondence pair  ================= #
    # Get each image row & column
    rows1, cols1 = cv_image1.shape[:2]
    rows2, cols2 = cv_image2.shape[:2]
    kptsC = np.array(kptsB.cpu()) + [cols1, 0]
        
    output_img = Image.new("RGB", (im1.width + im2.width, im1.height))
    output_img.paste(im1, (0, 0))
    output_img.paste(im2, (im1.width, 0))

    draw_correspondences(output_img, np.array(kptsA.cpu()), kptsC)


    pdb.set_trace()


    return kptsA, kptsB, warp, matches


        
