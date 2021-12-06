#include <fstream>
#include <ceres/ceres.h>
#include <Eigen/Dense>

#include "pose_local_parameterization.h"
#include "util.h"
#include "relative_pose_factor.h"

class PoseGraph {
  public:

    PoseGraph() = default;

    void add_pose(int pose_id, const double & timestamp, const Eigen::Matrix<double, 3,4> & pose) {
        std::vector<double> para_pose(7);
        Eigen::Map<Eigen::Vector3d> t(para_pose.data());
        t = pose.block<3,1>(0,3);
    
        Eigen::Map<Eigen::Quaterniond> q(para_pose.data() + 3);
        q = Eigen::Quaterniond(pose.block<3,3>(0,0));

        para_poses_.insert(std::make_pair(pose_id, para_pose));
        pose_timestamps_.push_back(std::make_pair(pose_id, timestamp));
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem_.AddParameterBlock(para_poses_.at(pose_id).data(), 7, local_parameterization);
    }

    void add_relative_pose_factor(int pose_id, int pose_id_second, const Eigen::Matrix<double,3,4> & rel_pose) {
        
        RelativePoseFactor * factor = new RelativePoseFactor(rel_pose);
        problem_.AddResidualBlock(factor, NULL, 
            std::vector<double*>{para_poses_.at(pose_id).data(), para_poses_.at(pose_id_second).data()});
    }

    void add_observation_factor(int pose_id, ceres::CostFunction * factor) {
        problem_.AddResidualBlock(factor, NULL, std::vector<double*>{para_poses_.at(pose_id).data()});
    }

    int find_nearest_pose_id(double timestamp) {
        for (int i = 0; i < pose_timestamps_.size() - 1; ++i) {
            if (pose_timestamps_.at(i).second <= timestamp && pose_timestamps_.at(i + 1).second >= timestamp) {
                return fabs(pose_timestamps_.at(i).second - timestamp) < 
                       fabs(pose_timestamps_.at(i + 1).second - timestamp) ?
                       pose_timestamps_.at(i).first : pose_timestamps_.at(i + 1).first;
            }
        }
        return -1;
    }

    void solve() {
        ceres::Solver::Options options;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 50;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem_, &summary);
    }

    void dump(std::string dump_path) {
        std::fstream fout(dump_path, std::ios::out);
        for (int i = 0; i < pose_timestamps_.size(); ++i) {
            int pose_id = pose_timestamps_.at(i).first;
            const std::vector<double> & para_pose = para_poses_.at(pose_id);
            fout << std::fixed << pose_timestamps_.at(i).second;
            for (int k = 0; k < 7; ++k) {
                fout << " " << para_pose.at(k);
            }
            fout << std::endl;
        }
    }

  private:
    std::unordered_map<int, std::vector<double> > para_poses_;
    std::vector<std::pair<int, double> > pose_timestamps_;
    ceres::Problem problem_;
};

