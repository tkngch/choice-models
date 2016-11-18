// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include "mvtnorm.h"

namespace MDFT {

/*
 * Multialternative Decision Field Theory as proposed by Roe, Busemeyer, and
 * Townsend (2001). The distance function is as specified by Hotaling,
 * Busemeyer, and Li (2010).
 *
 * At its heart, this is a C++ translation of Busemeyer's code
 * (http://mypage.iu.edu/~jbusemey/Dec_lect/Dec_prg/Analytic_2.txt). Though, I
 * changed the routine here and there.
 *
 * References
 *
 * Hotaling, J. M., Busemeyer, J. R., and Li, J. (2010). Theoretical
 * developments in decision field theory: Comment on Tsetsos, Usher, and Chater
 * (2010). Psychological Review, 117, 1294-1298.
 *
 * Roe, R. M., Busemeyer, J. R., and Townsend, J. T. (2001).  Multialternative
 * decision field theory: A dynamic connectionist model of decision making.
 * Psychological Review, 108, 370-392.
 *
 */

class Parameter {
public:
  Parameter(double wgt = 12, double phi1 = 0.022, double phi2 = 0.05,
            double sig2 = 0.05, double theta = 17.5, int stopping_time = 1001)
      : wgt(wgt), phi1(phi1), phi2(phi2), sig2(sig2), theta(theta),
        stopping_time(stopping_time){};

  ~Parameter(){};

  void print(void);
  std::string get_string(void);

  double wgt, phi1, phi2, sig2, theta, stopping_time;

}; // class Parameter

class Model {

public:
  Model(void){};
  ~Model(void){};

  // predicts choice probability given `stopping_time' external stopping rule.
  // parameter.theta is ignored here.
  void predict(std::vector<double> &choice_probability,
               const Eigen::MatrixXd &M, const Parameter &parameter);

  // predicts choice probability with `n_simulations' runs of simulations. A
  // run is terminated when one of the following conditions is met: (1) maximum
  // coefficient of preference vector reaches parameter.theta; or (2) the
  // number of preference update reaches `max_iter'. The simulation is slow, so
  // executing this method may take a long time. This ignores
  // parameter.stopping_time.
  void simulate(std::vector<double> &choice_probability,
                const Eigen::MatrixXd &alternatives, const Parameter &parameter,
                const unsigned int n_simulations = 1000,
                const unsigned int max_iter = 1000);

private:
  unsigned int n_alternatives, n_dimensions;

  void construct_feedback_matrix(Eigen::MatrixXd &S, const Eigen::MatrixXd &M,
                                 const Parameter &parameter);
  unsigned int check_feedback_matrix(Eigen::MatrixXd &S);
  void construct_contrast_matrix(Eigen::MatrixXd &C);
  void construct_weight_vector(Eigen::VectorXd &W);
  void construct_preference_and_covariance(
      Eigen::VectorXd &Eta, Eigen::MatrixXd &Omega, const Eigen::MatrixXd &C,
      const Eigen::MatrixXd &M, const Eigen::VectorXd &W,
      const Eigen::MatrixXd &S, const Parameter &parameter);
  void construct_L_matrix(Eigen::MatrixXd &L, const unsigned int i);
  double compute_choice_probability(const Eigen::MatrixXd &L,
                                    const Eigen::MatrixXd &Eta,
                                    const Eigen::MatrixXd &Omega);

  // computes multivariate normal cumulative density with libMvtnorm
  double multivariate_normal_cdf(const Eigen::VectorXd &Leta,
                                 const Eigen::MatrixXd &Lomega);

  int run(std::mt19937 &rng, const Eigen::MatrixXd &S, const Eigen::MatrixXd &C,
          const Eigen::MatrixXd &M, const Eigen::MatrixXd &W,
          const Parameter &parameter, const unsigned int max_iter);

}; // Model

}; // namespace MDFT
