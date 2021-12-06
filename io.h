#ifndef IO_H
#define IO_H

#include <Eigen/Dense>

void load_gtposes(const std::string & file_path, std::vector<std::pair<double, Eigen::Matrix<double, 3, 4>>>& gt_poses);

void load_gnss_observations(const std::string & file_path, std::vector<std::pair<double, Eigen::Vector3d>>& gnss_data);

#endif