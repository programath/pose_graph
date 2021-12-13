#include <vector>
#include <fstream>
#include <iostream>

#include "io.h"

std::vector<std::string> split(const std::string& str, const std::string& delim) {  
	std::vector<std::string> res;  
	if("" == str) return res;  
	char * strs = new char[str.length() + 1] ; //不要忘了  
	strcpy(strs, str.c_str());   
 
	char * d = new char[delim.length() + 1];  
	strcpy(d, delim.c_str());  
 
	char *p = strtok(strs, d);  
	while(p) {  
		std::string s = p; //分割得到的字符串转换为string类型  
		res.push_back(s); //存入结果数组  
		p = strtok(NULL, d);  
	}  
 
	return res;  
}

void load_gtposes(const std::string & file_path, std::vector<std::pair<double, Eigen::Matrix<double, 3, 4>>>& gt_poses) {
    std::fstream gt_poses_file;
    std::string line;
    gt_poses_file.open(file_path);
    std::cout << file_path << std::endl;
    std::vector<std::string> csv_coord;
    while (getline(gt_poses_file, line))
    {
        csv_coord = split(line, ",");
        // double qw, qx, qy, qz, tx, ty, tz;
        double timestamp = stod(csv_coord[0]);
        double gt_x = std::stod(csv_coord[1]);
        double gt_y = std::stod(csv_coord[2]);
        double gt_height = std::stod(csv_coord[3]);
        double gt_qx = std::stod(csv_coord[4]);
        double gt_qy = std::stod(csv_coord[5]);
        double gt_qz = std::stod(csv_coord[6]);
        double gt_qw = std::stod(csv_coord[7]);

        Eigen::Quaterniond G_R_V_gt(gt_qw, gt_qx, gt_qy, gt_qz);

        Eigen::Matrix<double,3,4> T;
        T.block<3, 3>(0, 0) = G_R_V_gt.toRotationMatrix();
        T.block<3, 1>(0, 3) = Eigen::Vector3d(gt_x, gt_y, gt_height);
        // T << p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11];
        gt_poses.push_back(std::make_pair(timestamp, T));
    }
}

void load_gnss_observations(const std::string & file_path, std::vector<std::pair<double, Eigen::Vector3d>>& gnss_data) {
    std::fstream gnsss_file;
    std::string line;
    gnsss_file.open(file_path);
    std::cout << file_path << std::endl;
    std::vector<std::string> csv_coord;
    while (getline(gnsss_file, line))
    {
        csv_coord = split(line, ",");
        // double qw, qx, qy, qz, tx, ty, tz;
        double timestamp = stod(csv_coord[0]);
        double gnss_x = std::stod(csv_coord[1]);
        double gnss_y = std::stod(csv_coord[2]);
        double gnss_z = std::stod(csv_coord[3]);

        gnss_data.push_back(std::make_pair(timestamp, Eigen::Vector3d(gnss_x, gnss_y, gnss_z)));
    }
}