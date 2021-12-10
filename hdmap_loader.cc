#include "hdmap_loader.h"

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

cv::Mat convert_to_mat(std::vector<Element> element_dict)
{
    std::vector<cv::Point2d> element_xy;
    for (int i=0; i<element_dict.size(); i++)
        for (int j=0; j<element_dict[i].xyz.size(); j++)
        {
            cv::Point2d xy (element_dict[i].xyz[j][0], element_dict[i].xyz[j][1]);
            element_xy.push_back(xy);
        }
    cv::Mat source = cv::Mat(element_xy).reshape(1);
    source.convertTo(source, CV_32F);
    return source;
}

HDMapDataBase::HDMapDataBase(std::string txt_path) {
    parser_hdmap(txt_path);
}

HDMapDataBase::HDMapDataBase(std::vector<std::string> txt_path) {
    for (int i=0; i<txt_path.size(); i++)
        parser_hdmap(txt_path[i]);
}

void HDMapDataBase::parser_hdmap(std::string txt_path) {
    std::cout << "Parsing HDMap from " << txt_path << ".." << std::endl;
    std::ifstream inFile(txt_path, std::ios::in);
    std::string x;
    std::vector<std::string> keys;
    std::vector<std::map<std::string, std::string>> element_data_;
    while (getline(inFile, x))
    {
        std::vector<std::string> line = string_split(x, ",");
        if (line[0] == "fid")
        {
            keys = line;
            continue; // skip head
        }
        std::map<std::string, std::string> data_;
        for (int i=0; i<line.size(); i++)
        {
            data_[keys[i]] = line[i];
//            std::cout << keys[i] << " " << line[i] << std::endl;
        }
        element_data_.push_back(data_);
    }
    std::cout << "Get " << element_data_.size() << " elements from " << txt_path << "." << std::endl << std::endl;
    std::vector<std::string> tmp = string_split(txt_path, "/.");
    std::string elem_type = tmp[tmp.size() - 3];
    hdmap_[elem_type] = element_data_;
}

void HDMapDataBase::get_element_from_hdmap(std::string element_key)
{
    std::vector<std::map<std::string, std::string>> element_dict_str_ = hdmap_[element_key_map_[element_key]];
    for (int i=0; i<element_dict_str_.size(); i++)
    {
        Element e;
        e.fid = stoi(element_dict_str_[i]["fid"]);
        std::vector<std::string> xyz_str = string_split(element_dict_str_[i]["xyz"], " ");
        for (int j=0; j<xyz_str.size(); j+=3)
        {
            Eigen::Vector3d xyz;
            xyz << std::stod(xyz_str[j]), std::stod(xyz_str[j+1]), std::stod(xyz_str[j+2]);
            e.xyz.push_back(xyz);
            element_idx_.push_back(i);
        }
        element_dict_.push_back(e);
    }
    build_kdtree();
}

void HDMapDataBase::build_kdtree() {
    cv::Mat source = convert_to_mat(element_dict_);
    cv::flann::KDTreeIndexParams indexParams(2);
    pkdtree_ = new cv::flann::Index(source, indexParams);
}

std::vector<Element> HDMapDataBase::find_neighbors(Eigen::Vector3d camera_xyz, double radius) {
    std::vector<Element> near_elem;
    for (int i=0; i<element_dict_.size(); i++) {
        for (int j=0; j<element_dict_[i].xyz.size(); j++) {
            double dst = (camera_xyz - element_dict_[i].xyz[j]).norm();
            if (dst <= radius) {
                near_elem.push_back(element_dict_[i]);
            }
        }
    }
    return near_elem;
}

std::vector<Element> HDMapDataBase::find_neighbors(Eigen::Vector3d camera_xyz, int query_topk) {
    std::vector<float> query {static_cast<float>(camera_xyz[0]), static_cast<float>(camera_xyz[1])};
    std::vector<int> retrieved_idx(query_topk);
    std::vector<float> retrieved_dst(query_topk);
    cv::flann::SearchParams params(32);
    pkdtree_->knnSearch(query, retrieved_idx, retrieved_dst, query_topk, params);
    std::vector<Element> near_elem;
    for (auto &r: retrieved_idx)
        near_elem.push_back(element_dict_[element_idx_[r]]);
    return near_elem;
}

double calc_dst_point_to_plane(Eigen::Vector3d camera_xyz, Eigen::Vector4d coeff) {
	double d = fabs((camera_xyz.dot(coeff.topRows(3))+ coeff[3]) / coeff.topRows(3).norm());
	return d;
}

// https://www.licc.tech/article?id=72
// ax+by+cz+d=0
Eigen::Vector4d plane_fitting(const std::vector<Eigen::Vector3d> & plane_pts) {
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

bool HDMapDataBase::construct_plane_height_constraint(Eigen::Vector3d camera_xyz, double radius,
                                                      double & camera_height, std::string element_key) {
    get_element_from_hdmap(element_key);
    std::vector<Element> near_elem = find_neighbors(camera_xyz, radius);

    std::vector<Eigen::Vector3d> near_xyz;
    for (auto & elem: near_elem)
        for (int i=0; i<elem.xyz.size(); i++)
            near_xyz.push_back(elem.xyz[i]);

    Eigen::Vector4d coeff = plane_fitting(near_xyz);
    camera_height = calc_dst_point_to_plane(camera_xyz, coeff);
    return true;
}

bool HDMapDataBase::construct_plane_height_constraint(Eigen::Vector3d camera_xyz, int query_topk,
                                                      double & camera_height, std::string element_key) {
    get_element_from_hdmap(element_key);
    std::vector<Element> near_elem = find_neighbors(camera_xyz, query_topk);

    std::vector<Eigen::Vector3d> near_xyz;
    for (auto & elem: near_elem)
        for (int i=0; i<elem.xyz.size(); i++)
            near_xyz.push_back(elem.xyz[i]);

    Eigen::Vector4d coeff = plane_fitting(near_xyz);
    camera_height = calc_dst_point_to_plane(camera_xyz, coeff);
    return true;
}

std::vector<Element> HDMapDataBase::get_element_dict() {
    return element_dict_;
}