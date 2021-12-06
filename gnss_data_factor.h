#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "util.h"

class GNSSDataFactor : public ceres::SizedCostFunction<2, 7>
{
    public:
        GNSSDataFactor(const Eigen::Vector3d &measure, const Eigen::Vector3d & sensor_gnss_to_body) : 
            measure_(measure), sensor_gnss_to_body_(sensor_gnss_to_body) {
            sqrt_info = 100.0;
        }

        virtual bool Evaluate(double const *const *parameteres, double *residuals, double **jacobians) const {

            Eigen::Vector3d Pi = Eigen::Vector3d(parameteres[0][0], parameteres[0][1], parameteres[0][2]);
            Eigen::Quaterniond Qi = Eigen::Quaterniond(parameteres[0][6], parameteres[0][3], parameteres[0][4], parameteres[0][5]);

            Eigen::Vector3d Pi_est = Qi * sensor_gnss_to_body_ + Pi;
            
            Eigen::Map<Eigen::Matrix<double, 2, 1> > residual(residuals);
            residual = sqrt_info * (measure_ - Pi_est).head<2>();

            if (jacobians) {
                Eigen::Matrix3d Ri = Qi.toRotationMatrix();
                
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor> > jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();
                    jacobian_pose_i.block<2, 3>(0, 0) = -Eigen::Matrix3d::Identity().block<2, 3>(0, 0);
                    jacobian_pose_i.block<2, 3>(0, 3) = (Ri * skewSymmetric(sensor_gnss_to_body_)).block<2, 3>(0, 0);
                } 
            }
            return true;
        }

        Eigen::Vector3d measure_;
        Eigen::Vector3d sensor_gnss_to_body_;
        double sqrt_info;
};
