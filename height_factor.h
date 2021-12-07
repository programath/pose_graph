#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "util.h"

class HeightFactor : public ceres::SizedCostFunction<1, 7>
{
    public:
        HeightFactor(const double &measure, const double & sensor_plane_to_body) : 
            measure_(measure), sensor_plane_to_body_(sensor_plane_to_body) {
            sqrt_info = 100.0;
        }

        virtual bool Evaluate(double const *const *parameteres, double *residuals, double **jacobians) const {

            Eigen::Vector3d Pi = Eigen::Vector3d(parameteres[0][0], parameteres[0][1], parameteres[0][2]);
            Eigen::Quaterniond Qi = Eigen::Quaterniond(parameteres[0][6], parameteres[0][3], parameteres[0][4], parameteres[0][5]);

            double Pi_est = Pi.z() - sensor_plane_to_body_;
            
            residuals[0] = sqrt_info * (measure_ - Pi_est);

            if (jacobians) {
                Eigen::Matrix3d Ri = Qi.toRotationMatrix();
                
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor> > jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();
                    jacobian_pose_i(0, 2) = -1;
                    jacobian_pose_i = sqrt_info * jacobian_pose_i;
                } 
            }
            return true;
        }

        double measure_;
        double sensor_plane_to_body_;
        double sqrt_info;
};
