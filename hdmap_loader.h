#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <cmath>
#include <iomanip>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>

typedef struct
{
    int fid;
    std::vector<Eigen::Vector3d> xyz;
} Element;

typedef struct
{
    int node_id;
    std::vector<int> inlier_elem;
    Eigen::Vector3d center;
    int parent_id;
    std::vector<int> subnodes_id;
} Node;

class HDMapDataBase {
  
  public:

    HDMapDataBase() = delete;

    HDMapDataBase(std::string hdmap_path);

    HDMapDataBase(std::vector<std::string> hdmap_paths);

    void get_element_from_hdmap(std::string element_key);

    std::vector<Element> get_element_dict();

    std::vector<Element> find_neighbors(Eigen::Vector3d camera_xyz, double radius);

    std::vector<Element> find_neighbors(Eigen::Vector3d camera_xyz, int query_topk);

    bool construct_plane_height_constraint(Eigen::Vector3d camera_xyz, double radius, double & height, std::string element_key);

    bool construct_plane_height_constraint(Eigen::Vector3d camera_xyz, int query_topk, double & camera_height, std::string element_key);
  private:
    std::map<std::string, std::vector<std::map<std::string, std::string>>> hdmap_;
    std::vector<Element> element_dict_;
    std::vector<int> element_idx_;
    cv::flann::Index* pkdtree_;
    std::map<std::string, std::string> element_key_map_ = {
            {"lane", "100101-bxdp_line"},
            {"arrow", "100109-bxdp_jt"},
            {"lg", "200102-hldp_lg"},
            {"lightpole", "400104-zmlddp"},
            {"xsbz", "400107-jcjdp_xsbz"},
            {"roadside", "600110-lysdp"},
            {"md", "600111-md"},
            {"camera", "800101-sxtdp"}
    };

    void parser_hdmap(std::string txt_path);

    void build_kdtree();

};
