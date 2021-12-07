#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <cmath>
#include <iomanip>
#include </usr/include/eigen3/Eigen/Dense>
#include <math.h>

using namespace std;
using namespace Eigen;


typedef struct
{
    int id;
    vector<Vector3d> xyz;
} Lane;

typedef struct 
{
    double timestamp;
    Vector3d xyz;
    Matrix3d rot;
} Pose;


vector<string> string_split(const string &str, const char *delim)
{
	vector <std::string> strlist;
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

void load_lane_from_hdmap(string txt_path, vector<Lane> *lane_dict)
{   
    ifstream inFile(txt_path, ios::in);
    string x;
    while (getline(inFile, x))
    {
        Lane lane;
        vector<string> line = string_split(x, ",");
        if (line[0] == "fid(T)")
            continue; // skip head
        lane.id = stoi(line[0]);
        vector<string> xyz_str = string_split(line[1], " ");
        for (int i=0; i<xyz_str.size(); i+=3){
            // cout << stod(xyz_str[i]) << " " << stod(xyz_str[i+1]) << " " << stod(xyz_str[i+2]) << endl;
            Vector3d xyz;
            xyz << stod(xyz_str[i]), stod(xyz_str[i+1]), stod(xyz_str[i+2]); 
            lane.xyz.push_back(xyz);
        }
        // cout << lane.id << " " << lane.xyz.size() << endl;
        lane_dict->push_back(lane);
    }
}

void get_xyz_from_lane(vector<Lane> lane_dict, vector<Vector3d> *xyz)
{
    for (int i=0; i<lane_dict.size(); i++)
        for (int j=0; j<lane_dict[i].xyz.size(); j++)
        {
            xyz->push_back(lane_dict[i].xyz[j]);
        }
}

vector<Vector3d> find_neighbors(Vector3d camera_xyz, vector<Vector3d> lane_xyz, double radius)
{
    vector<Vector3d> near_xyz;
    for (int i=0; i<lane_xyz.size(); i++)
    {
        double dst = (camera_xyz - lane_xyz[i]).norm();
        if (dst <= radius)
        {
            near_xyz.push_back(lane_xyz[i]);
        }
    }
    return near_xyz;
}

// https://www.licc.tech/article?id=72
// ax+by+cz+d=0
Vector4d plane_fitting(const vector<Vector3d> & plane_pts) 
{
    Vector3d center = Eigen::Vector3d::Zero();
    for (const auto & pt : plane_pts) center += pt;
    center /= plane_pts.size();

    MatrixXd A(plane_pts.size(), 3);
    for (int i = 0; i < plane_pts.size(); i++) {
        A(i, 0) = plane_pts[i][0] - center[0];
        A(i, 1) = plane_pts[i][1] - center[1];
        A(i, 2) = plane_pts[i][2] - center[2];
    }

    JacobiSVD<MatrixXd> svd(A, ComputeThinV);
    const float a = svd.matrixV()(0, 2);
    const float b = svd.matrixV()(1, 2);
    const float c = svd.matrixV()(2, 2);
    const float d = -(a * center[0] + b * center[1] + c * center[2]);
    return Vector4d(a, b, c, d);
}

double calc_dst_point_to_plane(Vector3d camera_xyz, Vector4d coeff)
{
	double d = fabs((camera_xyz.dot(coeff.topRows(3))+ coeff[3]) / coeff.topRows(3).norm());
	return d;
}


void transform_from_lidar_to_camera(Pose lidar_pose, Pose * camera_pose)
{
    Matrix4d camera_to_lidar;
    camera_to_lidar << 0.0264, -0.0089, -0.9996, -0.50,
                    0.9995, -0.0176, 0.0266, 0.06,
                    -0.0178, -0.9998, 0.0085, 0,
                    0, 0, 0, 1;
    camera_pose->xyz << lidar_pose.rot * camera_to_lidar.col(3).topRows(3) + lidar_pose.xyz;
}


int main()
{
    vector<Lane> lane_dict;
    vector<Vector3d> all_xyz, near_xyz;
    string lane_hdmap_txt = "/home/jinyu/Documents/HDMapProjector/data/map/100101-bxdp_line.utm.txt";
    double radius = 250.0;
    
    load_lane_from_hdmap(lane_hdmap_txt, &lane_dict);
    cout << "Loaded " << lane_dict.size() << " lane from " << lane_hdmap_txt << endl;
    get_xyz_from_lane(lane_dict, &all_xyz);
    cout << "Loaded " << all_xyz.size() << " Points from " << lane_hdmap_txt << endl;

    ifstream inFile("/home/jinyu/Documents/HDMapProjector/data/poses/global_lidar_pose.csv", ios::in);
    string x;
    while (getline(inFile, x))
    {
        Pose lidar_pose, camera_pose;
        vector<string> line = string_split(x, ",");
        lidar_pose.timestamp = stod(line[0]);
        camera_pose.timestamp = stod(line[0]);
        lidar_pose.xyz << stod(line[1]), stod(line[2]), stod(line[3]);
        Quaterniond quat(stod(line[4]), stod(line[5]), stod(line[6]), stod(line[7]));
        lidar_pose.rot = quat.toRotationMatrix();
        // cout << "TimeStamp: " << camera_pose.timestamp << ", Position: (" << lidar_pose.xyz.transpose() << ")" << endl;
        // cout << lidar_pose.rot << endl;

        transform_from_lidar_to_camera(lidar_pose, &camera_pose);
        cout << "TimeStamp: " << camera_pose.timestamp << ", Position: (" << camera_pose.xyz.transpose() << ")" << endl;
        
        near_xyz = find_neighbors(camera_pose.xyz, all_xyz, radius);
        if (near_xyz.size() <= 20)
            continue;
        else
            cout << near_xyz.size() << " neighbors are found." << endl;
        
        Vector4d coefficient;
        coefficient = plane_fitting(near_xyz);
        for (int i=0; i<4; i++)
            cout << coefficient[i] << " ";
        cout << endl;
        double hd = calc_dst_point_to_plane(camera_pose.xyz, coefficient);
        cout << hd << endl << endl;

    }

    return 0;
}