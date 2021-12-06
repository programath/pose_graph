#ifndef RELATIVE_POSE_FACTOR_H
#define RELATIVE_POSE_FACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "util.h"

class RelativePoseFactor : public ceres::SizedCostFunction<6, 7, 7>
{
    public:
        RelativePoseFactor(const Eigen::Matrix<double, 3, 4> & rel_pose) {
            rel_t_ = rel_pose.block<3, 1>(0, 3);
            rel_rot_ = Eigen::Quaterniond(rel_pose.block<3, 3>(0, 0));
            sqrt_info = 100.0;
        }

        virtual bool Evaluate(double const *const *parameteres, double *residuals, double **jacobians) const {

            Eigen::Vector3d Pi = Eigen::Vector3d(parameteres[0][0], parameteres[0][1], parameteres[0][2]);
            Eigen::Quaterniond Qi = Eigen::Quaterniond(parameteres[0][6], parameteres[0][3], parameteres[0][4], parameteres[0][5]);
            Eigen::Vector3d Pj = Eigen::Vector3d(parameteres[1][0], parameteres[1][1], parameteres[1][2]);
            Eigen::Quaterniond Qj = Eigen::Quaterniond(parameteres[1][6], parameteres[1][3], parameteres[1][4], parameteres[1][5]);

            // Ti^1 * Tj = Rel_pose --> Rel_pose^-1 * Ti^-1 * Tj = err.  (Ti*Rel_pose)^-1 * Tj*[1, 1/2\theta] = 2[0 I]((Ti*Rel_pose)^-1 * Tj)_L * [0, 1/2I]
            // Ti^1 * Tj = Rel_pose --> Rel_pose^-1 * Ti^-1 * Tj = err.  (Ti*[1,1/2\theta]*Rel_pose)^-1 * Tj 
            // = -[0, I] * ((Tj*-1 * Ti*[1,1/2\theta]*Rel_pose)) = -2[0 I] * (Tj*-1 * Ti)_L * (Rel_pose)R * [0, 1/2I] 
            Eigen::Quaterniond Qj_est = Qi * rel_rot_;
            Eigen::Vector3d Pj_est = Qi * rel_t_ + Pi;

            Eigen::Quaterniond err = Qj_est.inverse() * Qj;
            
            Eigen::Map<Eigen::Matrix<double, 6, 1> > residual(residuals);
            residual.block<3, 1>(0, 0) = Pj - Pj_est;
            if (err.w() > 0)
                residual.block<3, 1>(3, 0) = 2.0 * err.vec();
            else
                residual.block<3, 1>(3, 0) = -2.0 * err.vec();

            residual = sqrt_info * residual;

            if (jacobians) {
                Eigen::Matrix3d Ri = Qi.toRotationMatrix();
                Eigen::Matrix3d Rj = Qj.toRotationMatrix();
                
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();
                    jacobian_pose_i.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
                    jacobian_pose_i.block<3, 3>(0, 3) = Ri * skewSymmetric(rel_t_);
                    jacobian_pose_i.block<3, 3>(3, 3) = Qleft(Qj_est).block<3,3>(1,1);
                } 
                if (jacobians[1]) {
                    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > jacobian_pose_j(jacobians[1]); 
                    jacobian_pose_j.setZero();
                    jacobian_pose_j.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                    jacobian_pose_j.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
                    jacobian_pose_j.block<3, 3>(3, 3) = -(Qleft(Qj.inverse() * Qi) * Qright(rel_rot_)).block<3,3>(1,1);
                }
            }
            return true;
        }

        Eigen::Vector3d rel_t_;
        Eigen::Quaterniond rel_rot_;
        double sqrt_info;
};

#endif