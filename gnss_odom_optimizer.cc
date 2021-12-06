#include <vector>
#include <Eigen/Dense>

#include "pose_graph.h"
#include "io.h"
#include "relative_pose_factor.h"
#include "gnss_data_factor.h"

int main() {

    std::vector<std::pair<double, Eigen::Matrix<double, 3, 4> > > gt_poses;
    std::vector<std::pair<double, Eigen::Vector3d> > gnss_data;

    std::string pose_file = "/data/cs55/mapping/2021-11-29/bag5/data/global_camera_pose.csv";
    std::string gnss_file = "/data/cs55/mapping/2021-11-29/bag5/data/gnss_measure.csv";

    load_gtposes(pose_file, gt_poses);
    load_gnss_observations(gnss_file, gnss_data);

    PoseGraph pose_graph;

    for (int id = 0; id < gt_poses.size(); ++id) {
        double timestamp = gt_poses[id].first;
        Eigen::Matrix<double, 3, 4> pose = gt_poses[id].second;
        pose_graph.add_pose(id, timestamp, pose);
        if (id > 0) {
            Eigen::Matrix<double, 3, 4> rel_pose;
            rel_pose.block<3,3>(0,0) = 
                gt_poses.at(id - 1).second.block<3,3>(0,0).transpose() * pose.block<3,3>(0,0);
            rel_pose.block<3,1>(0,3) = 
                gt_poses.at(id - 1).second.block<3,3>(0,0).transpose() * (pose.block<3,1>(0,3) - gt_poses.at(id - 1).second.block<3,1>(0,3));

            pose_graph.add_relative_pose_factor(id - 1, id, rel_pose);
        }   
    }     

    Eigen::Vector3d sensor_gps_to_body = Eigen::Vector3d(0, 0, 0);   

    for (int idx = 0; idx < gnss_data.size(); ++idx) {
        double timestamp = gnss_data.at(idx).first;
        int pose_id = pose_graph.find_nearest_pose_id(timestamp);
        if (pose_id < 0)
            continue;
        GNSSDataFactor * factor = new GNSSDataFactor(gnss_data.at(idx).second, sensor_gps_to_body);
        pose_graph.add_observation_factor(pose_id, factor);
    }

    pose_graph.solve();

    pose_graph.dump("solved_poses.csv");
    return true;
}