// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)
/*
 * decision_by_sampling.hpp
 *
 * Probability to evaluate depends on similarity between alternatives in a
 * choice set, while probability of winning a comparison assumes all the
 * samples are equally likely compared.
 *
 * This is work in progress. Please do not redstribute this code or any
 * variants/modifications.
 */

// #define NDEBUG   // disable assert

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <exception>
#include <iostream>
#include <map>
#include <math.h>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace DbS {

const double MAX_STATES = 500;

class Parameter {
public:
  Parameter(double similarity_slope = 3.0, double recognition_intercept = 0.1,
            double recognition_slope = -50, double threshold = 0.1);

  double similarity_slope;
  double recognition_intercept;
  double recognition_slope;
  double threshold;

  void print(void) const;
  std::string get_string(void) const;
  bool check_values(void) const;
};

class Model {
public:
  Model(void){};
  ~Model(void){};

  // single choice set without memory samples
  void predict(std::vector<double> &choice_probability,
               const Eigen::MatrixXd &choice_set, const Parameter &parameter);

  // single choice set without memory samples. additionally computes the
  // expected number of comparisons
  void predict(std::vector<double> &choice_probability,
               double &expected_number_of_comparisons,
               const Eigen::MatrixXd &choice_set, const Parameter &parameter);

  // single choice set with memory samples
  void predict(std::vector<double> &choice_probability,
               const Eigen::MatrixXd &choice_set,
               const std::vector<Eigen::VectorXd> &memory_samples,
               const Parameter &parameter);

  // single choice set with memory samples. additionally computes the expected
  // number of comparisons.
  void predict(std::vector<double> &choice_probability,
               double &expected_number_of_comparisons,
               const Eigen::MatrixXd &choice_set,
               const std::vector<Eigen::VectorXd> &memory_samples,
               const Parameter &parameter);

private:
  unsigned int n_dimensions;
  unsigned int n_alternatives;

  // set = alternatives in choice set + memory samples
  std::vector<Eigen::VectorXd> set;

  std::vector<Eigen::MatrixXd> distance;
  std::vector<Eigen::MatrixXd> similarity;
  Eigen::MatrixXd p_evaluation;
  Eigen::MatrixXd p_win;
  Eigen::VectorXd accumulation_rate;

  // build distance and similarity matrices, computes p_evaluation and p_win,
  // and calculates accumulation_rate
  void compute(const Eigen::MatrixXd &choice_set,
               const std::vector<Eigen::VectorXd> &memory_samples,
               const Parameter &parameter);

  void build_distance_and_similarity_matrices(const double &similarity_slope);
  void build_evaluation_probability(void);
  void build_winning_probability(const double &recognition_intercept,
                                 const double &recognition_slope);
  void compute_accumulation_rate(void);
};

class Walk {
public:
  Walk(const unsigned int n_alternatives, const double threshold)
      : n_alternatives(n_alternatives), threshold(threshold),
        boundary(static_cast<int>(ceil(n_alternatives * threshold))){};

  ~Walk(void){};

  void start(std::vector<double> &choice_probability,
             const Eigen::VectorXd &p_step);

  void start(std::vector<double> &choice_probability,
             double &expected_number_of_steps, const Eigen::VectorXd &p_step);

private:
  const unsigned int n_alternatives;
  const double threshold;
  const int boundary;
  unsigned int n_transient_states;
  std::map<std::vector<int>, unsigned int> transient_state_index;
  Eigen::MatrixXd Q;      // transition probability within transient states
  Eigen::MatrixXd R;      // transition probability to absorbing states
  Eigen::MatrixXd Z;      // one hot vector to represent the starting point
  Eigen::MatrixXd ZIQinv; // Z * (I - Q)^{-1}
  Eigen::MatrixXd P;      // probability to reach absorbing states

  void get_ready(const Eigen::VectorXd &p_step);

  void fill_transient_state_index(void);
  void compute_range(std::vector<int> &range);
  unsigned int compute_n_states(const unsigned int n,
                                const std::vector<int> &range);
  bool update_state(std::vector<int> &state, const std::vector<int> &range,
                    const unsigned int n);

  void fill_matrices(const Eigen::VectorXd &p_step);
};
}
