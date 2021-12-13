import numpy as np
import matplotlib.pyplot as plt


class PosePlotter(object):
    
    def __init__(self, gt_txt, gnss_txt):
        self.gt_trajectory = {}
        self.gnss_measure = {}

        self.load_gt_trajectory(gt_txt)
        print('=> load %d ground-truth pose from %s' % (len(self.gt_trajectory.keys()), gt_txt))
        self.load_measurement(gnss_txt)
        print('=> load %d measurement from %s' % (len(self.gnss_measure.keys()), gnss_txt))


    def load_gt_trajectory(self, gt_txt):
        with open(gt_txt, 'r') as f:
            lines = f.readlines()
            for l in lines:
                data = l.strip().split(',') # ts, x, y, z, qx, qy, qz, qw
                assert len(data) == 8
                self.gt_trajectory[eval(data[0])] = [eval(data[1]), eval(data[2]), eval(data[3])]

    def load_measurement(self, gnss_txt):
        with open(gnss_txt, 'r') as f:
            lines = f.readlines()
            for l in lines:
                data = l.strip().split(',') # ts, x, y, z,
                assert len(data) == 4
                self.gnss_measure[eval(data[0])] = [eval(data[1]), eval(data[2]), eval(data[3])]

    def show_in_3d_map(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        gt_xyz = np.asarray(list(self.gt_trajectory.values()))
        gnss_xyz = np.asarray(list(self.gnss_measure.values()))
        # print(gt_xyz.shape, gnss_xyz.shape)
        ax.scatter(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], c='b', marker='o')
        ax.scatter(gnss_xyz[:, 0], gnss_xyz[:, 1], gnss_xyz[:, 2], c='r', marker='s')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        # ax.axis('equal')
        plt.show()
    
    def show_in_2d_map(self):
        fig = plt.figure()
        gt_xyz = np.asarray(list(self.gt_trajectory.values()))
        gnss_xyz = np.asarray(list(self.gnss_measure.values()))
        # print(gt_xyz.shape, gnss_xyz.shape)
        plt.plot(gt_xyz[:, 0], gt_xyz[:, 1], c='b', marker='o', markersize=5)
        plt.plot(gnss_xyz[:, 0], gnss_xyz[:, 1], c='r', marker='s', markersize=1)
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.show()
    
    def track_gnss_measurement_in_2d_map(self, radius=50):
        def find_neighbors(query, sources, radius):
            dst = np.linalg.norm(query-sources, ord=2, axis=1)
            neighbors_idx = dst < radius
            neighbors = sources[neighbors_idx]
            return neighbors
        
        gt_xyz = np.asarray(list(self.gt_trajectory.values()))
        gnss_xyz = np.asarray(list(self.gnss_measure.values()))
        for query in gnss_xyz:
            neighbors = find_neighbors(query, gt_xyz, radius)
            plt.plot(gt_xyz[:, 0], gt_xyz[:, 1], c='g', marker='o', markersize=5)
            plt.plot(neighbors[:, 0], neighbors[:, 1], c='b', marker='o', markersize=5)
            plt.plot(query[0], query[1], c='r', marker='*', markersize=5)
            plt.xlabel('X Axis')
            plt.ylabel('Y Axis')
            plt.pause(0.5)
            



if __name__ == '__main__':
    # plotter = PosePlotter('../data/global_camera_pose.csv', '../data/gnss_measure.csv')
    plotter = PosePlotter('../build/solved_poses.csv', '../data/gnss_measure.csv')
    plotter.show_in_3d_map() # show all the ground-truth trajectory and GNSS observations in 3D map
    plotter.show_in_2d_map() # show all the ground-truth trajectory and GNSS observations in 2D map 
    plotter.track_gnss_measurement_in_2d_map() # show one GNSS observation and its nearby ground-truth trajectory in 2D map
 