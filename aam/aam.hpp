// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include <iostream>
#include <algorithm>
#include <vector>
#include <eigen3/Eigen/Dense>

namespace AAM {

class Parameter {
public:
  Parameter(double a0 = 10, double alpha = 0.5, double e = 0.05, double d = 0.8)
      : a0(a0), alpha(alpha), e(e), d(d){};

  ~Parameter(){};

  void print(void);
  std::string get_string(void);

  double a0, alpha, e, d;

}; // class Parameter

class Model {

public:
  Model(const unsigned int n_simulations = 2000,
        const unsigned int n_iterations = 100)
      : n_simulations(n_simulations), n_iterations(n_iterations){};

  ~Model(){};

  // estimates choice probability. this runs n_simulations runs of simulations,
  // and it's rather slow.
  void predict(std::vector<double> &choice_probability,
               const Eigen::MatrixXd &alternatives, const Parameter &parameter);

  // computes expected preference. This doesn't rely on simulations, so it's
  // faster than predict, but expected preference does not necessarily
  // correlate with choice probability.
  void compute(std::vector<double> &expected_preference,
               const Eigen::MatrixXd &alternatives, const Parameter &parameter);

private:
  unsigned int n_simulations, n_iterations;
  unsigned int n_alternatives, n_dimensions;

  void construct_feedback_matrix(Eigen::MatrixXd &S,
                                 const Parameter &parameter);
  void compute_attribute_accessibility(Eigen::VectorXd &W,
                                       const Eigen::MatrixXd &M,
                                       const Parameter &parameter);
  void compute_attribute_values(Eigen::MatrixXd &V, const Eigen::MatrixXd &M,
                                const Parameter &parameter);

  int simulate(std::mt19937 &rng, const Eigen::MatrixXd &S,
               const Eigen::MatrixXd &W, const Eigen::MatrixXd &V,
               const Parameter &parameter);

  int find_winner(Eigen::MatrixXd &P);

}; // Model

}; // namespace AAM
