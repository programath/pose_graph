import os
import sys
import numpy as np
import cv2

from pose_database import PoseDataBase
import open3d as o3d

lane_category = 0
pole_category = 1

def load_pcd_file(pcd_file):
    pcd = o3d.io.read_point_cloud(pcd_file)
    return np.asarray(pcd.points)

def load_timestamps(timestamp_file):
    with open(timestamp_file, "r") as f:
        timestamps0 = f.readlines()
        timestamps = np.array([eval(t.strip()) for t in timestamps0])
        timestampe_str = [t.strip() for t in timestamps0]
    return timestamps, timestampe_str

def get_camera_info(camera_info):
    camera_info["image_height"] = 1080 # 930
    camera_info["image_width"] = 1920
    camera_info["K"] = np.array([[1053.1000, 0., 985.26410],
                                 [0., 1053.1000, 551.74212],
                                 [0., 0., 1.]])
    camera_info["distortion"] = np.array([0., 0., 0., 0., 0.])
    camera_info["rect"] = np.eye(3)
    camera_info["sensor_camera_to_lidar"] = np.array([[0.0264, -0.0089, -0.9996, -0.50],
                                                      [0.9995, -0.0176, 0.0266, 0.06],
                                                      [-0.0178, -0.9998, 0.0085, 0],
                                                      [0, 0, 0, 1]])


def project_hdmap_on_image(hdmap, pose, camera_info, image=None, load_keys=['lane']):
    # camera parametric model
    img_h = camera_info["image_height"]
    img_w = camera_info["image_width"]
    K = camera_info["K"]
    distortion = camera_info["distortion"]
    rect = camera_info["rect"]

    pose_R = pose[:3, :3] # rotation_cam_to_world

    plot_width = 15

    if image is None:
        image = np.zeros([img_h, img_w], dtype="uint8")
    mask = np.ones([img_h, img_w], dtype="uint8") * 255

    for k in load_keys:
        values = hdmap[k]

        hdmap_ids = []
        hdmap_masks = []
        hdmap_categories = []

        color_gray = np.random.randint(0, 175)
        color_rgb = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

        for idx, coordinates in values.items():
            image_cooridnate = (coordinates - pose[:3, 3]).dot(pose_R)
            image_cooridnate = image_cooridnate[(image_cooridnate[:, 2] > 1) & (image_cooridnate[:, 2] < 50)]

            image_cooridnate = np.divide(image_cooridnate, image_cooridnate[:, 2:])
            if image_cooridnate.shape[0] >= 2:
                projected_points, _ = cv2.projectPoints(image_cooridnate,          \
                                                    np.zeros(3, dtype=np.float32), \
                                                    np.zeros(3, dtype=np.float32), \
                                                    K,                             \
                                                    distortion)
                projected_points = projected_points.squeeze(1)
                projected_points = projected_points[(projected_points[:, 0] > 0) & (projected_points[:, 0] < img_w) \
                    & (projected_points[:, 1] > 0) & (projected_points[:,1] < img_h)]     
                if projected_points.shape[0] >= 2:
                    # print(projected_points)
                    projected_points = projected_points[None]
                    # projected_points = projected_points.reshape([1, -1, 2])
                    projected_points = projected_points.astype(np.int32)
                    cv2.polylines(mask, projected_points, False, color_gray, plot_width)
                    cv2.polylines(image, projected_points, False, color_rgb, 5)
                    cv2.putText(image, str(idx), (projected_points[0][0][0], projected_points[0][0][1]), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 0), 1)
                    hdmap_masks.append(mask)
                    hdmap_categories.append(lane_category)
                    hdmap_ids.append(idx)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    show_board = np.concatenate([mask, image], axis=0)
    show_board = cv2.resize(show_board, (0,0), fx=0.3, fy=0.3)
    cv2.imshow("hdmap", show_board)
    cv2.waitKey(0)

def project_pointcloud_on_image(points, pose, camera_info, image=None):
    # camera parametric model
    img_h = camera_info["image_height"]
    img_w = camera_info["image_width"]
    K = camera_info["K"]
    distortion = camera_info["distortion"]
    rect = camera_info["rect"]

    pose_R = pose[:3, :3] # rotation_cam_to_world

    mask = np.zeros([img_h, img_w], dtype="uint8")

    coordinate_index = np.arange(points.shape[0])
    image_cooridnate = (points - pose[:3, 3]).dot(pose_R)
    coordinate_index = coordinate_index[(image_cooridnate[:, 2] > 1) & (image_cooridnate[:, 2] < 100)]
    image_cooridnate = image_cooridnate[(image_cooridnate[:, 2] > 1) & (image_cooridnate[:, 2] < 100)]

    image_cooridnate = np.divide(image_cooridnate, image_cooridnate[:, 2:])
    if image_cooridnate.shape[0] >= 2:
        projected_points, _ = cv2.projectPoints(image_cooridnate,              \
                                                np.zeros(3, dtype=np.float32), \
                                                np.zeros(3, dtype=np.float32), \
                                                K,                             \
                                                distortion)
        projected_points = projected_points.squeeze(1)
        projected_points = projected_points[(projected_points[:, 0] > 0) & (projected_points[:, 0] < img_w) & (projected_points[:, 1] > 0) & (projected_points[:,1] < img_h)]                                      
        if projected_points.shape[0] >= 2:
            projected_points = projected_points[None]
            # projected_points = projected_points.reshape([1, -1, 2])
            projected_points = projected_points.astype(np.int32)
            for p in projected_points[0]:
                cv2.circle(mask, tuple(p), 5, (255, 255, 255), -1)
                cv2.circle(image, tuple(p), 5, (255, 255, 0), -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    show_board = np.concatenate([mask, image], axis=0)
    show_board = cv2.resize(show_board, (0,0), fx=0.3, fy=0.3)
    cv2.imshow("hdmap", show_board)
    cv2.waitKey(0)


if __name__ == "__main__":

    databag_path = "./data"

    if len(sys.argv) == 2:
        databag_path = sys.argv[1]

    image_list_path = os.path.join(databag_path, "list.txt")
    pose_list_path = os.path.join(databag_path, "poses", "global_lidar_poses.csv")
    image_folder_path = os.path.join(databag_path, "images")
    map_folder_path = os.path.join(databag_path, "map", "data_tsinghua.csv")
    zero_folder_path = os.path.join(databag_path, "poses", "zero_utm")

    timestamps, timestamps_str = load_timestamps(image_list_path)
    pose_database = PoseDataBase(pose_list_path)
    timestamps, timestamps_str = pose_database.filter_timestamps(timestamps, timestamps_str)

    camera_info = {}
    get_camera_info(camera_info)
    
    poses = pose_database.query_poses(timestamps, extrinsic=camera_info["sensor_camera_to_lidar"])
    images = [os.path.join(image_folder_path, '%s.jpg' % x) for x in timestamps_str]

    zero_utm = np.loadtxt(zero_folder_path)
    # simple hdmap loader
    with open(map_folder_path, "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
        
    points = []
    for l in lines:
        data = l.split(',')
        points.append([eval(data[0]), eval(data[1]), eval(data[2])])
    points = np.array(points)
    points = points - zero_utm

    for pose, im_file in zip(poses, images): # pose: pose_lidar_to_world
        sensor_camera_to_lidar = camera_info["sensor_camera_to_lidar"]
        pose_cam_to_world = pose
        if im_file == "/data/cs55/mapping/2021-11-29/bag1/images/1638174096.210925817.jpg":

            image = cv2.imread(im_file, 1)

            lidar_points = load_pcd_file("/data/cs55/mapping/2021-11-29/bag1/scliosam/Scans/000008.pcd")
            lidar_pose = pose_database.query_poses([1638174096.181149])[0]
            print(lidar_pose, pose_cam_to_world)

            lidar_pose_current = pose_database.query_poses([1638174096.210925817])[0]
            pose_lidar_to_curr = np.eye(4).astype(np.float)
            pose_lidar_to_curr[:3, :3] = (lidar_pose_current[:3, :3].T).dot(lidar_pose[:3, :3])
            pose_lidar_to_curr[:3, 3:] = (lidar_pose_current[:3, :3].T).dot(lidar_pose[:3, 3:] - lidar_pose_current[:3, 3:])

            lidar_points = lidar_points.dot(pose_lidar_to_curr[:3, :3].T) + pose_lidar_to_curr[:3, 3]

            project_pointcloud_on_image(lidar_points, camera_info["sensor_camera_to_lidar"], camera_info, image)