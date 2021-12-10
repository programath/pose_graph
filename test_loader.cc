#include <vector>
#include <Eigen/Dense>

#include "chrono"
#include "hdmap_loader.h"

int main() {
    // Uncomment to enable the plane constraint.
    std::vector<std::string> hdmap_paths {
        "/home/jinyu/Documents/pose_graph/data/100101-bxdp_line.utm.txt",
        "/home/jinyu/Documents/pose_graph/data/100109-bxdp_jt.utm.txt"
    };
    HDMapDataBase hdmap_database(hdmap_paths);
    hdmap_database.get_element_from_hdmap("lane");
    auto element_dict = hdmap_database.get_element_dict();
    Element test_sample = element_dict[701];
    Eigen::Vector3d test_xyz(3);
    test_xyz << test_sample.xyz[0], test_sample.xyz[1], test_sample.xyz[2];
    std::cout << "TEST: " << test_xyz.transpose() << " " << test_sample.fid << std::endl;
    auto t1 = std::chrono::steady_clock::now();
    auto neighbors = hdmap_database.find_neighbors(test_xyz, 5);
    auto t2 = std::chrono::steady_clock::now();
    for (auto &n: neighbors)
        std::cout << "FIND: " << n.xyz[0].transpose() << " " << n.fid << std::endl;
    double totaltime = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "Time: " << totaltime << std::endl;
    std::cout << std::endl;

    std::cout << "TEST: " << test_xyz.transpose() << " " << test_sample.fid << std::endl;
    auto t3 = std::chrono::steady_clock::now();
    auto neighbors2 = hdmap_database.find_neighbors(test_xyz, 2.5);
    auto t4 = std::chrono::steady_clock::now();
    for (auto &n: neighbors2)
        std::cout << "FIND: " << n.xyz[0].transpose() << " " << n.fid << std::endl;
    double totaltime2 = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::cout << "Time: " << totaltime2 << std::endl;
    return 0;
}