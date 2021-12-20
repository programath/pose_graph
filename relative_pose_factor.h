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
            sqrt_info << 100.0, 100, 100, 1000, 1000, 1000;
        }

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

            Eigen::Vector3d Pi = Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
            Eigen::Quaterniond Qi = Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
            Eigen::Vector3d Pj = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Quaterniond Qj = Eigen::Quaterniond(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

            // Ti^1 * Tj = Rel_pose --> Rel_pose^-1 * Ti^-1 * Tj = err.  (Ti*Rel_pose)^-1 * Tj*[1, 1/2\theta] = 2[0 I]((Ti*Rel_pose)^-1 * Tj)_L * [0, 1/2I]
            // Ti^1 * Tj = Rel_pose --> Rel_pose^-1 * Ti^-1 * Tj = err.  (Ti*[1,1/2\theta]*Rel_pose)^-1 * Tj 
            // = -[0, I] * ((Tj*-1 * Ti*[1,1/2\theta]*Rel_pose)) = -2[0 I] * (Tj*-1 * Ti)_L * (Rel_pose)R * [0, 1/2I] 
            Eigen::Quaterniond Qj_est = Qi * rel_rot_;
            Eigen::Vector3d Pj_est = Qi * rel_t_ + Pi; // FIXME: remove the rotation part

            Eigen::Quaterniond err = Qj_est.inverse() * Qj;
            
            Eigen::Map<Eigen::Matrix<double, 6, 1> > residual(residuals);
            residual.block<3, 1>(0, 0) = Pj - Pj_est;
            if (err.w() > 0)
                residual.block<3, 1>(3, 0) = 2.0 * err.vec();
            else
                residual.block<3, 1>(3, 0) = -2.0 * err.vec();

            residual = sqrt_info.asDiagonal() * residual;

            if (jacobians) {
                Eigen::Matrix3d Ri = Qi.toRotationMatrix();
                Eigen::Matrix3d Rj = Qj.toRotationMatrix();
                
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > jacobian_pose_i(jacobians[0]);
                    jacobian_pose_i.setZero();
                    jacobian_pose_i.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
                    jacobian_pose_i.block<3, 3>(0, 3) = Ri * skewSymmetric(rel_t_);
                    jacobian_pose_i.block<3, 3>(3, 3) = -(Qleft(Qj.inverse() * Qi) * Qright(rel_rot_)).block<3,3>(1,1);
                    jacobian_pose_i = sqrt_info.asDiagonal() * jacobian_pose_i;
                } 
                if (jacobians[1]) {
                    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > jacobian_pose_j(jacobians[1]); 
                    jacobian_pose_j.setZero();
                    jacobian_pose_j.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                    jacobian_pose_j.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
                    jacobian_pose_j.block<3, 3>(3, 3) = Qleft(err).block<3,3>(1,1);
                    jacobian_pose_j = sqrt_info.asDiagonal() * jacobian_pose_j;
                }
            }
            return true;
        }

        void check(double const *const *parameters) {

            double * res = new double [6];
            double ** jaco = new double * [2];
            jaco[0] = new double [6 * 7];
            jaco[1] = new double [6 * 7];
            Evaluate(parameters, res, jaco);

            std::cout << "my" << std::endl;
            std::cout << Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> >(jaco[0]) << std::endl;
            std::cout << Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> >(jaco[1]) << std::endl;
            
            Eigen::Vector3d Pi = Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
            Eigen::Quaterniond Qi = Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
            Eigen::Vector3d Pj = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Quaterniond Qj = Eigen::Quaterniond(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

            // Ti^1 * Tj = Rel_pose --> Rel_pose^-1 * Ti^-1 * Tj = err.  (Ti*Rel_pose)^-1 * Tj*[1, 1/2\theta] = 2[0 I]((Ti*Rel_pose)^-1 * Tj)_L * [0, 1/2I]
            // Ti^1 * Tj = Rel_pose --> Rel_pose^-1 * Ti^-1 * Tj = err.  (Ti*[1,1/2\theta]*Rel_pose)^-1 * Tj 
            // = -[0, I] * ((Tj*-1 * Ti*[1,1/2\theta]*Rel_pose)) = -2[0 I] * (Tj*-1 * Ti)_L * (Rel_pose)R * [0, 1/2I] 
            Eigen::Quaterniond Qj_est = Qi * rel_rot_;
            Eigen::Vector3d Pj_est = Qi * rel_t_ + Pi;

            Eigen::Quaterniond err = Qj_est.inverse() * Qj;
            
            Eigen::Matrix<double, 6, 1> residual;
            residual.block<3, 1>(0, 0) = Pj - Pj_est;
            if (err.w() > 0)
                residual.block<3, 1>(3, 0) = 2.0 * err.vec();
            else
                residual.block<3, 1>(3, 0) = -2.0 * err.vec();

            residual = sqrt_info.asDiagonal() * residual;

            const double eps = 1e-6;
            Eigen::Matrix<double, 6, 12> num_jacobian;
            for (int k = 0; k < 12; ++k) {
                Eigen::Vector3d Pi = Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]);
                Eigen::Quaterniond Qi = Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
                Eigen::Vector3d Pj = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]);
                Eigen::Quaterniond Qj = Eigen::Quaterniond(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

                int a = k / 3, b = k % 3;
                Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;
                if (a == 0) 
                    Pi += delta;
                else if (a == 1)
                    Qi = Qi * deltaQ(delta);
                else if (a == 2)
                    Pj += delta;
                else if (a == 3)
                    Qj = Qj * deltaQ(delta);

                Eigen::Quaterniond Qj_est = Qi * rel_rot_;
                Eigen::Vector3d Pj_est = Qi * rel_t_ + Pi;

                Eigen::Quaterniond err = Qj_est.inverse() * Qj;
                
                Eigen::Matrix<double, 6, 1> tmp_residual;
                tmp_residual.block<3, 1>(0, 0) = Pj - Pj_est;
                if (err.w() > 0)
                    tmp_residual.block<3, 1>(3, 0) = 2.0 * err.vec();
                else
                    tmp_residual.block<3, 1>(3, 0) = -2.0 * err.vec();

                tmp_residual = sqrt_info.asDiagonal() * tmp_residual;
                num_jacobian.col(k) = (tmp_residual - residual) / eps;
            }
            std::cout << "num" << std::endl;
            std::cout << num_jacobian << std::endl;
        }

        Eigen::Vector3d rel_t_;
        Eigen::Quaterniond rel_rot_;
        Eigen::Vector<double, 6, 1> sqrt_info;
};

#endif