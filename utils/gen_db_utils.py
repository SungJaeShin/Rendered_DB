import os
import pdb
import numpy as np
from PIL import Image

def get_image_id(img_path):
    return os.path.splitext(os.path.basename(img_path))[0].lstrip('0')

def find_index(params, cur_img_idx):
    for i in range(len(params)):
        if not params[i].strip():
            continue

        if params[i].split()[0] == cur_img_idx:
            return i

# camParam format : image_id, camera_model, width, height, ?, ?, ?, ?
# gtPose format : image_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name
def make_render_dataset(model, data_path, transforms=None):
    camParam = os.path.join(data_path, 'cameras.txt')
    gtPose = os.path.join(data_path, 'images.txt')
    if not os.path.exists(camParam) or not os.path.exists(gtPose):
        raise FileNotFoundError("cameras.txt or images.txt not found in the specified directory.")

    imgs = [os.path.join(root, file) for root, _, files in os.walk(os.path.join(data_path, 'images/')) for file in files]
    if len(imgs) == 0:
        raise FileNotFoundError("Images not found in the specified directory.")

    render_dataset = []
    with open(camParam, 'r') as cam_file, open(gtPose, 'r') as pose_file:
        cam_params = cam_file.readlines() # cam_params[0].split()[0] -> 이러면 첫 id 반환
        gt_poses = pose_file.readlines()

        for i in range(len(imgs)):
            cur_img_id = get_image_id(imgs[i])
            cam_idx = find_index(cam_params, cur_img_id)
            gt_idx = find_index(gt_poses, cur_img_id)
            # print(f"cur idx: {i}, cur img idx: {cur_img_id}, cam idx: {cam_idx}, gt idx: {gt_idx}")

            cur_pil_img = Image.open(imgs[i])
            cur_tf_pil_img = transforms(cur_pil_img).unsqueeze(dim=0)
            cur_img_vlad = model(cur_tf_pil_img).detach().numpy()
            cur_img = np.array(cur_pil_img)

            render_dataset.append({
                'idx': i,
                'image': cur_img,
                'vlad': cur_img_vlad,
                'gt_pose': np.array(gt_poses[gt_idx].split()[1:8], dtype=np.float32),
                'cam_type': cam_params[cam_idx].split()[1],
                'cam_params': np.array(cam_params[cam_idx].split()[2:], dtype=np.float32)
            })

    return render_dataset