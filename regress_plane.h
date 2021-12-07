#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <math.h>

typedef struct
{
    int id;
    std::vector<Eigen::Vector3d> xyz;
} Lane;

typedef struct 
{
    double timestamp;
    Eigen::Vector3d xyz;
    Eigen::Matrix3d rot;
} Pose;

class HDMapDataBase {
  
  public:

    HDMapDataBase() = delete;

    HDMapDataBase(std::string hdmap_path);

    void load_lane_from_hdmap(std::string txt_path);

    void get_xyz_from_lane(std::vector<Eigen::Vector3d> & xyz);

    std::vector<Eigen::Vector3d> find_neighbors(Eigen::Vector3d camera_xyz, double radius);

    bool construct_plane_height_constraint(Eigen::Vector3d camera_xyz, double radius, double & height);

  private:
    std::vector<Lane> lane_dict_;
};
