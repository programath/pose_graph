import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy import interpolate

class PoseDataBase:
    def __init__(self, gt_poses_file):
        self.load_gt_poses(gt_poses_file)
        

    def load_gt_poses(self, gt_poses_path):
        
        with open(gt_poses_path, "r") as f:
            poses = f.readlines()
            poses = [p.strip() for p in poses]
        
        qs, ts, key_times = [], [], []
        for p in poses:
            p = p.split(' ')
            timestamp = float(p[0])
            qs.append(np.array([float(x) for x in p[4:8]]))
            ts.append(np.array([float(x) for x in p[1:4]]))
            key_times.append(timestamp)
        
        qs = np.array(qs)
        self.rots = Rotation.from_quat(qs)
        self.ts = np.array(ts)
        self.key_times = np.array(key_times)
        self.rot_slerp = Slerp(key_times, self.rots)
        self.t_interp = interpolate.interp1d(key_times, self.ts, axis=0)

    def query_poses(self, timestamps, extrinsic=None):
        interp_rots = self.rot_slerp(timestamps)
        interp_ts = self.t_interp(timestamps).reshape([-1, 3, 1])
        interp_rots = interp_rots.as_matrix()
        if extrinsic is None:
            return np.concatenate([interp_rots, interp_ts], axis=2)
        else:
            return np.concatenate([interp_rots, interp_ts], axis=2).dot(extrinsic)

    def filter_timestamps(self, timestamps, timestamps_str):
        max_t = np.max(self.key_times)
        min_t = np.min(self.key_times)
        timestamps = timestamps[(timestamps > min_t) & (timestamps < max_t)]
        index = (timestamps > min_t) & (timestamps < max_t)
        timestamps_str = [timestamps_str[i] for i in range(len(index)) if index[i]]
        return timestamps, timestamps_str
