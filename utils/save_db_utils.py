import h5py

def save_render_dataset(db, save_path):
    with h5py.File(save_path, 'a') as h5:
        for idx in range(len(db)):
            h5.create_group(str(idx))
            h5.create_dataset(str(idx) + '/image', data=db[idx]['image'], compression="gzip")
            h5.create_dataset(str(idx) + '/vlad', data=db[idx]['vlad'], compression="gzip")
            h5.create_dataset(str(idx) + '/gt_pose', data=db[idx]['gt_pose'], compression="gzip")
            h5.create_dataset(str(idx) + '/cam_type', data=db[idx]['cam_type']) # only save byte type
            h5.create_dataset(str(idx) + '/cam_params', data=db[idx]['cam_params'], compression="gzip")
