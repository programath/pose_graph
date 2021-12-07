#include "regress_plane.h"

std::vector<std::string> string_split(const std::string &str, const char *delim)
{
	std::vector <std::string> strlist;
	int size = str.size();
	char *input = new char[size+1];
	strcpy(input, str.c_str());
	char *token = std::strtok(input, delim);
	while (token != NULL) {
		strlist.push_back(token);
		token = std::strtok(NULL, delim);
	}
	delete []input;
	return strlist;
}

HDMapDataBase::HDMapDataBase(std::string txt_path) {
    load_lane_from_hdmap(txt_path);
}

void HDMapDataBase::load_lane_from_hdmap(std::string txt_path)
{   
    std::ifstream inFile(txt_path, std::ios::in);
    std::string x;
    while (getline(inFile, x))
    {
        Lane lane;
        std::vector<std::string> line = string_split(x, ",");
        if (line[0] == "fid(T)")
            continue; // skip head
        lane.id = stoi(line[0]);
        std::vector<std::string> xyz_str = string_split(line[1], " ");
        for (int i=0; i<xyz_str.size(); i+=3){
            // cout << stod(xyz_str[i]) << " " << stod(xyz_str[i+1]) << " " << stod(xyz_str[i+2]) << endl;
            Eigen::Vector3d xyz;
            xyz << std::stod(xyz_str[i]), std::stod(xyz_str[i+1]), std::stod(xyz_str[i+2]); 
            lane.xyz.push_back(xyz);
        }
        // cout << lane.id << " " << lane.xyz.size() << endl;
        lane_dict_.push_back(lane);
    }
}

void HDMapDataBase::get_xyz_from_lane(std::vector<Eigen::Vector3d> & xyz)
{
    xyz.clear();
    for (int i=0; i<lane_dict_.size(); i++)
        for (int j=0; j<lane_dict_[i].xyz.size(); j++)
        {
            xyz.push_back(lane_dict_[i].xyz[j]);
        }
}

std::vector<Eigen::Vector3d> HDMapDataBase::find_neighbors(Eigen::Vector3d camera_xyz, double radius)
{
    std::vector<Eigen::Vector3d> near_xyz;
    for (int i=0; i<lane_dict_.size(); i++) {
        for (int j=0; j<lane_dict_[i].xyz.size(); j++) {
            double dst = (camera_xyz - lane_dict_[i].xyz[j]).norm();
            if (dst <= radius) {
                near_xyz.push_back(lane_dict_[i].xyz[j]);
            }
        }
    }
    return near_xyz;
}

double calc_dst_point_to_plane(Eigen::Vector3d camera_xyz, Eigen::Vector4d coeff)
{
	double d = fabs((camera_xyz.dot(coeff.topRows(3))+ coeff[3]) / coeff.topRows(3).norm());
	return d;
}

// https://www.licc.tech/article?id=72
// ax+by+cz+d=0
Eigen::Vector4d plane_fitting(const std::vector<Eigen::Vector3d> & plane_pts) 
{
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    for (const auto & pt : plane_pts) center += pt;
    center /= plane_pts.size();

    Eigen::MatrixXd A(plane_pts.size(), 3);
    for (int i = 0; i < plane_pts.size(); i++) {
        A(i, 0) = plane_pts[i][0] - center[0];
        A(i, 1) = plane_pts[i][1] - center[1];
        A(i, 2) = plane_pts[i][2] - center[2];
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
    const float a = svd.matrixV()(0, 2);
    const float b = svd.matrixV()(1, 2);
    const float c = svd.matrixV()(2, 2);
    const float d = -(a * center[0] + b * center[1] + c * center[2]);
    return Eigen::Vector4d(a, b, c, d);
}

bool HDMapDataBase::construct_plane_height_constraint(Eigen::Vector3d camera_xyz, double radius, double & camera_height) {
    std::vector<Eigen::Vector3d> near_xyz = find_neighbors(camera_xyz, radius);
    Eigen::Vector4d coeff = plane_fitting(near_xyz);
    camera_height = calc_dst_point_to_plane(camera_xyz, coeff);
    return true;
}