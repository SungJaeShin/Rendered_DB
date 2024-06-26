import os
import torch
import h5py
import faiss
import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw

from utils.visualize_utils import *
from models.LightGlue import LightGlue

import pdb

# Find N Candidates and Distances
def find_distance(db):
    vlad1 = db[0]['vlad']
    vlad2 = db[1]['vlad']
    vlad3 = db[2]['vlad']

    tmp = []
    tmp.append(vlad1)
    tmp.append(vlad2)
    tmp = np.array(tmp)
    tmp = tmp.reshape(tmp.shape[0], tmp.shape[2])

    index = faiss.IndexFlatL2(tmp.shape[1])
    index.is_trained # Output: True

    index.add(tmp)
    cand_num = 1

    cand_dist, cand_index = index.search(vlad3, cand_num) 
    print(f"candidate index: {cand_index}, candidate distance: {cand_dist}")

    return cand_dist, cand_index

# Filtered the best candidate image using local descriptor
# SURF is the best method for extracting local descriptor, but that method is patented 
def calculate_score(query_img, query_kpt, query_des, 
                    cand_img, cand_kpt, cand_des, threshold=0.95):
    # Using BF Matcher and KNN Matching Method
    bf = cv.BFMatcher()

    # Get best features
    matches = bf.knnMatch(query_des, cand_des, k=2)
    good1 = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good1.append([m])

    matches = bf.knnMatch(cand_des, query_des, k=2)
    good2 = []
    for m,n in matches:
        if m.distance < threshold*n.distance:
            good2.append([m])

    # Get Final best matching results of features
    final_good = []
    for i in good1:
        img1_id1 = i[0].queryIdx
        img2_id1 = i[0].trainIdx

        (x1,y1) = query_kpt[img1_id1].pt
        (x2,y2) = cand_kpt[img2_id1].pt

        for j in good2:
            img1_id2 = j[0].queryIdx
            img2_id2 = j[0].trainIdx

            (a1,b1) = cand_kpt[img1_id2].pt
            (a2,b2) = query_kpt[img2_id2].pt

            if (a1 == x2 and b1 == y2) and (a2 == x1 and b2 == y1):
                final_good.append(i)        

    print(f"# of Correspondence btw query and cand: {len(final_good)}")
    if len(final_good) == 0:
        return 0

    ### If you want to plot the result then erase annotation code 
    # img_matching_result = cv.drawMatchesKnn(query_img, query_kpt, cand_img, cand_kpt, final_good, None, [0,0,255],flags=2)
    # save_path = '/home/sj/workspace/paper/iccas2024/results/[1] matching_imgs/matching_results.png'
    # cv.imwrite(save_path, img_matching_result)

    return len(final_good)

# Calculate Relative Pose btw Query and Candidate

# ===================== [ToDo] =====================
# Not Finish!!!
def lightglue_matcher(imgA, imgB, kptsA, kptsB, desA, desB, config, device):
    # features => SuperPoint, DISK, SIFT, ALIKED
    matcher = LightGlue(features='superpoint', weight_path=config['lightglue_method']['superpoint_feat']).eval().cuda()  

    # Change Keypoints Dim
    img1_kpt = kptsA.reshape(1, kptsA.shape[0], kptsA.shape[1])
    img2_kpt = kptsB.reshape(1, kptsB.shape[0], kptsB.shape[1])
    
    # Change Descriptors Dim
    img1_des = desA.T
    img1_des = img1_des.reshape(1, img1_des.shape[0], img1_des.shape[1])
    img2_des = desB.T
    img2_des = img2_des.reshape(1, img2_des.shape[0], img2_des.shape[1])

    # Change Images Dim
    img1 = np.transpose(imgA, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    img1 = img1[np.newaxis, ...]  # (C, H, W) -> (1, C, H, W)
    img2 = np.transpose(imgB, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    img2 = img2[np.newaxis, ...]  # (C, H, W) -> (1, C, H, W)

    # Convert to torch.Tensor
    img1_kpt = torch.tensor(img1_kpt).float().cuda()
    img2_kpt = torch.tensor(img2_kpt).float().cuda()
    img1_des = torch.tensor(img1_des).float().cuda()
    img2_des = torch.tensor(img2_des).float().cuda()
    img1 = torch.tensor(img1).float().cuda()
    img2 = torch.tensor(img2).float().cuda()

    # Make dict format
    feat_img1 = {'keypoints': img1_kpt, 'descriptors': img1_des, 'image': img1}
    feat_img2 = {'keypoints': img2_kpt, 'descriptors': img2_des, 'image': img2}

    # Matcher output => matches0 matches1 matching_scores0 matching_scores1 stop matches scores
    matches01 = matcher({'image0': feat_img1, 'image1': feat_img2})
    # kptsA, kptsB, matches01 = [rbd(x) for x in [feat_img1, feat_img2, matches01]]  # remove batch dimension
    matches = matches01['matches']  # indices with shape (K,2)
    
    # Get Inlier Features 
    list_matches = [tensor.cpu().numpy() for tensor in matches]
    np_matches = np.concatenate(list_matches, axis=0)
    pointsA = kptsA[np_matches[..., 0]]  # coordinates in image #0, shape (K,2)
    pointsB = kptsB[np_matches[..., 1]]  # coordinates in image #1, shape (K,2)

    # # Not Finish!!!
    # imgA_pil = Image.fromarray(imgA)
    # draw_keypoints_on_image(imgA_pil, pointsA)
    # imgB_pil = Image.fromarray(imgB)
    # draw_keypoints_on_image(imgB_pil, pointsB)

    # pdb.set_trace()

