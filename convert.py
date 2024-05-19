
import os
import time
import torch
import h5py
# import rospy
import faiss

import cv2 as cv
import numpy as np
from queue import Queue
from PIL import Image as pil_Image
from cv_bridge import CvBridge, CvBridgeError

# ROS Messeage Related
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Applied NetVLAD
from utils.args_utils import *
from utils.gen_db_utils import make_render_dataset
from utils.save_utils import save_render_dataset
from utils.matching_utils import find_distance, calculate_score
from utils.extractor_utils import feature_extractor, descriptor_extractor, roma_based_extractor
from dataset.transforms import *
from models.NetVLAD import NetVLAD

import pdb

if __name__ == '__main__':
    device = 'cuda'
    opt = argument_parser()
    config = import_yaml(opt.config)

    # [Part 1: Get Pre-trained NetVLAD Model]
    # ===========================================================
    print("\033[1;37m========== Get Pre-trained NetVLAD Model ==========\033[0m")
    start_get_ptmodel = time.time()
    model = NetVLAD(config)
    model.load_state_dict(torch.load(config['pt_netvlad_path'])['state_dict'])
    print(f"Get Pre-trained NetVLAD Model: {(time.time() - start_get_ptmodel)}")

    # [Part 2: Make Render Dataset including Render Image and Camera Parameters]
    # ===========================================================
    print("\033[1;37m========== Render Image and Camera Parameters ==========\033[0m")
    start_get_cam_params = time.time()
    db = make_render_dataset(model, config['render_dataset_path'], transforms=T_KAIST)
    print(f"Gen Render Database: {(time.time() - start_get_cam_params)}")

    # [Part 3: Save HDF5 files in local path]
    # ===========================================================
    if config['save_db']:
        print("\033[1;37m========== Save HDF5 Database ==========\033[0m")
        start_save_db = time.time()
        save_render_dataset(db, config['save_db_path'])
        print(f"Save Render Database: {(time.time() - start_save_db)}")

    # [Part 4: Find the nearest Distance btw VLAD vectors]
    # ===========================================================
    print("\033[1;37m========== Calculate VLAD Distance ==========\033[0m")
    start_cal_dist = time.time()
    cand_dist, cand_index = find_distance(db)
    print(f"Calculate VLAD Distance: {(time.time() - start_cal_dist)}")

    # [Part 5: Find Feature Matching Method]
    # ===========================================================
    img1 = db[1]['image']
    img2 = db[2]['image']
    if len(img1.shape) == 3:
        img1 = cvt_rgb_to_gray(img1)
    if len(img2.shape) == 3:
        img2 = cvt_rgb_to_gray(img2)


    print(config['descriptor_method']['DAISY'])


    pdb.set_trace()

    # [Part 6: Change VINS-Mono Template using ROS]
    # ===========================================================


    


      