// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include "dbs.hpp"

using namespace DbS;

Parameter::Parameter(double similarity_slope, double recognition_intercept,
                     double recognition_slope, double threshold)
    : similarity_slope(similarity_slope),
      recognition_intercept(recognition_intercept),
      recognition_slope(recognition_slope), threshold(threshold) {

  // arbitrary limit the range. only up to 1e8.
  if (recognition_slope < -1e8) {
    recognition_slope = recognition_slope + 1e8;
  }
};

bool Parameter::check_values(void) const {
  if (similarity_slope < 0) {
    throw std::logic_error("Invalid parameter values: similarity_slope cannot be negative.");
  } else if (recognition_intercept < 0) {
    throw std::logic_error("Invalid parameter values: recognition_intercept cannot be negative.");
  } else if (recognition_slope > 0) {
    throw std::logic_error("Invalid parameter values: recognition_slope cannot be positive.");
  }
  return true;
  // NOTE
  // When recognition_intercept is too large (typically > 1), any difference
  // will not be recognised and choice probability tends to be all zero.
};

void Parameter::print(void) const { std::cout << get_string() << std::endl; };

std::string Parameter::get_string(void) const {
  std::stringstream ss;
  ss << "\"[" << similarity_slope << ", " << recognition_intercept << ", "
     << recognition_slope << ", " << threshold << "]\"";
  return ss.str();
};

void Model::predict(std::vector<double> &choice_probability,
                    const Eigen::MatrixXd &choice_set,
                    const Parameter &parameter) {

  // set memory_sample to empty vector
  std::vector<Eigen::VectorXd> memory_samples;
  memory_samples.resize(choice_set.cols());

  predict(choice_probability, choice_set, memory_samples, parameter);
}

void Model::predict(std::vector<double> &choice_probability,
                    double &expected_number_of_comparisons,
                    const Eigen::MatrixXd &choice_set,
                    const Parameter &parameter) {

  // set memory_sample to empty vector
  std::vector<Eigen::VectorXd> memory_samples;
  memory_samples.resize(choice_set.cols());

  predict(choice_probability, expected_number_of_comparisons, choice_set,
          memory_samples, parameter);
}

void Model::predict(std::vector<double> &choice_probability,
                    const Eigen::MatrixXd &choice_set,
                    const std::vector<Eigen::VectorXd> &memory_samples,
                    const Parameter &parameter) {

  // set expected_number_of_comparisons to be NaN
  double expected_number_of_comparisons =
      std::numeric_limits<double>::quiet_NaN();
  predict(choice_probability, expected_number_of_comparisons, choice_set,
          memory_samples, parameter);
}

void Model::predict(std::vector<double> &choice_probability,
                    double &expected_number_of_comparisons,
                    const Eigen::MatrixXd &choice_set,
                    const std::vector<Eigen::VectorXd> &memory_samples,
                    const Parameter &parameter) {

  // expected_number_of_comparisons is computed IF it is not set to NaN.

  compute(choice_set, memory_samples, parameter);

  choice_probability.resize(n_alternatives);

  if (!std::isnan(expected_number_of_comparisons)) {
    expected_number_of_comparisons = 0;
  }

  if (parameter.threshold == 0) {

    std::fill(begin(choice_probability), end(choice_probability),
              1.0 / n_alternatives);

  } else if (n_alternatives == 1) {

    choice_probability.at(0) = accumulation_rate(0);

  } else {

    Walk walk(n_alternatives, parameter.threshold);

    if (std::isnan(expected_number_of_comparisons)) {

      walk.start(choice_probability, accumulation_rate);

    } else {

      walk.start(choice_probability, expected_number_of_comparisons,
                 accumulation_rate);
    }
  }
}

void Model::compute(const Eigen::MatrixXd &choice_set,
                    const std::vector<Eigen::VectorXd> &memory_samples,
                    const Parameter &parameter) {

  parameter.check_values();

  n_alternatives = choice_set.rows();
  n_dimensions = choice_set.cols();
  assert(memory_samples.size() == n_dimensions);

  set.resize(n_dimensions);
  for (unsigned int dim = 0; dim < n_dimensions; dim++) {
    set.at(dim).resize(n_alternatives + memory_samples.at(dim).size());
    set.at(dim) << choice_set.col(dim), memory_samples.at(dim);
  }

  build_distance_and_similarity_matrices(parameter.similarity_slope);
  build_evaluation_probability();
  build_winning_probability(parameter.recognition_intercept,
                            parameter.recognition_slope);
  compute_accumulation_rate();

  if (accumulation_rate.maxCoeff() <= 1e-8) {
    throw std::logic_error("Accumulation rates are all zero.");
  }
}

void Model::build_distance_and_similarity_matrices(
    const double &similarity_slope) {

  // assert(similarity_slope >= 0);

  distance.resize(n_dimensions);
  similarity.resize(n_dimensions);

  for (unsigned int dim = 0; dim < n_dimensions; dim++) {

    distance.at(dim).resize(n_alternatives, set.at(dim).size());
    distance.at(dim).setZero();
    similarity.at(dim).resize(n_alternatives, set.at(dim).size());
    similarity.at(dim).setZero();

    for (unsigned int i = 0; i < n_alternatives; i++) {

      for (unsigned int j = 0; j < set.at(dim).size(); j++) {
        if (i == j || std::isnan(set.at(dim)(j)) ||
            std::isnan(set.at(dim)(i))) {
          continue;
        }

        distance.at(dim)(i, j) =
            fabs(set.at(dim)(i) - set.at(dim)(j)) / fabs(set.at(dim)(j));
        similarity.at(dim)(i, j) =
            exp(-1 * distance.at(dim)(i, j) * similarity_slope);
      }
    }
    // std::cout << "similarity at " << dim << std::endl;
    // std::cout << similarity.at(dim) << std::endl;
  }
}

void Model::build_evaluation_probability(void) {

  p_evaluation.resize(n_alternatives, n_dimensions);
  p_evaluation.setZero();

  double normaliser = 0;
  for (unsigned int dim = 0; dim < n_dimensions; dim++) {

    normaliser += similarity.at(dim).sum();

    for (unsigned int i = 0; i < n_alternatives; i++) {

      p_evaluation(i, dim) = similarity.at(dim).row(i).sum();
    }
  }

  if (normaliser > 0) {
    p_evaluation /= normaliser;
  } else {
    p_evaluation.setOnes();
    p_evaluation /= n_alternatives * n_dimensions;
  }

  // std::cout << "p_evaluation" << std::endl;
  // std::cout << p_evaluation << std::endl;
  // std::cout << "normaliser: " << normaliser << std::endl;
}

void Model::build_winning_probability(const double &recognition_intercept,
                                      const double &recognition_slope) {

  // assert(recognition_slope <= 0);

  auto logistic = [&recognition_intercept, &recognition_slope](const double x) {
    return 1.0 / (1.0 + exp((x - recognition_intercept) * recognition_slope));
  };

  p_win.resize(n_alternatives, n_dimensions);
  p_win.setZero();

  double p_comparison;
  int n_attributes;

  for (unsigned int dim = 0; dim < n_dimensions; dim++) {
    // subtract 1 from number of comparisons, so that there is no
    // self-comparison
    n_attributes = 0;
    for (unsigned int i = 0; i < set.at(dim).size(); i++) {
      if (!std::isnan(set.at(dim)(i))) {
        n_attributes++;
      }
    }
    p_comparison = 1.0 / (n_attributes - 1.0);

    for (unsigned int i = 0; i < n_alternatives; i++) {
      for (unsigned int j = 0; j < set.at(dim).size(); j++) {

        if (!std::isnan(set.at(dim)(i)) && set.at(dim)(i) > set.at(dim)(j)) {

          p_win(i, dim) += logistic(distance.at(dim)(i, j)) * p_comparison;
        }
      }
    }
  }
}

void Model::compute_accumulation_rate(void) {

  Eigen::MatrixXd tmp = p_evaluation.cwiseProduct(p_win);
  accumulation_rate.resize(n_alternatives);

  for (unsigned int i = 0; i < n_alternatives; i++) {
    accumulation_rate(i) = tmp.row(i).sum();
  }

  double sum = accumulation_rate.sum();
  if (sum > 1.0) {
    std::cout << "p_evaluation:" << std::endl;
    std::cout << p_evaluation << std::endl;
    std::cout << "p_win:" << std::endl;
    std::cout << p_win << std::endl;
    std::cout << "accumulation_rate:" << std::endl;
    std::cout << accumulation_rate << std::endl;
    throw std::runtime_error("Sum of accumulation_rate is more than 1.0.");
  }

  // assert(0 < sum); // sum can be 0 for multiple reasons:
  // (1). p_win is non-zero only when p_evaluation is zero. For example, p_win
  // = [[0, 1], [1, 0]] and p_evaluation = [[0.5, 0], [0.5, 0]].
  // (2). p_win is all zero.
  // Catch these conditions in the calling function.
}

void Walk::start(std::vector<double> &choice_probability,
                 const Eigen::VectorXd &p_step) {

  get_ready(p_step);

  choice_probability.resize(n_alternatives);
  std::fill(begin(choice_probability), end(choice_probability), 0.0);
  for (unsigned int i = 0; i < n_alternatives; i++) {
    choice_probability.at(i) = P(0, i);
  }
}

void Walk::start(std::vector<double> &choice_probability,
                 double &expected_number_of_steps,
                 const Eigen::VectorXd &p_step) {

  start(choice_probability, p_step);

  Eigen::MatrixXd One(n_transient_states, 1);
  One.setOnes();

  Eigen::MatrixXd res = ZIQinv * One;
  expected_number_of_steps = res(0, 0);
}

void Walk::get_ready(const Eigen::VectorXd &p_step) {

  assert(n_alternatives == p_step.size());

  fill_transient_state_index();

  Q.resize(n_transient_states, n_transient_states);
  R.resize(n_transient_states, n_alternatives);
  Z.resize(1, n_transient_states);
  fill_matrices(p_step);

  Eigen::MatrixXd I(n_transient_states, n_transient_states);
  I.setIdentity();

  Eigen::MatrixXd IQ = I - Q;
  Eigen::MatrixXd IQinv;
  if (n_transient_states < 1000) {
    // feasible only for a small matrix
    IQinv = IQ.inverse();
  } else {
    IQinv = IQ.ldlt().solve(I);
    double relative_error = (IQ * IQinv - I).norm() / I.norm();
    if (relative_error > 1e-8) {
      std::cout << "The relative error is too large: " << relative_error
                << std::endl;
      throw std::logic_error("The error is too large");
    }
  }

  ZIQinv = Z * IQinv;
  P = ZIQinv * R;
  assert(P.cols() == n_alternatives);
  assert(P.rows() == 1);
  assert(fabs(P.sum() - 1.0) < 1e-10);
}

void Walk::fill_transient_state_index(void) {

  std::vector<int> range;
  compute_range(range);

  n_transient_states = compute_n_states(n_alternatives, range);
  if (n_transient_states > MAX_STATES) {
    // don't even try to compute. It'd take too long or too errorneous.
    std::cout << "State space is too large: " << n_transient_states
              << std::endl;
    throw std::logic_error("State space is too large");
  }

  std::vector<int> state;
  state.resize(n_alternatives);
  std::fill(begin(state), end(state), range.at(0));

  transient_state_index.clear();

  unsigned int i = 0;
  while (update_state(state, range, n_alternatives)) {

    transient_state_index[state] = i++;
  }

  assert(n_transient_states == i);
}

void Walk::compute_range(std::vector<int> &range) {

  // computes the range the relative evidence (accumulated evidence - mean) can
  // take.  endpoint exclusive for transient states

  range.resize(2);
  range.at(0) = -1 * (boundary + (n_alternatives - 2) * (boundary - 1));
  range.at(1) = boundary;
}

unsigned int Walk::compute_n_states(const unsigned int n,
                                    const std::vector<int> &range) {
  unsigned int num = 1;
  unsigned int den = 1;

  unsigned int tmp = range.at(1) - range.at(0) - 1;
  for (unsigned int i = 0; i <= n - 2; i++) {
    num *= tmp + i;
  }
  for (unsigned int i = 1; i <= n - 1; i++) {
    den *= i;
  }

  return num / den;
}

bool Walk::update_state(std::vector<int> &state, const std::vector<int> &range,
                        const unsigned int n) {

  bool updated = false;

  for (unsigned int i = 0; i < n - 1; i++) {

    if (updated) {
      break;
    } else if (state.at(i) + 1 < range.at(1)) {
      state.at(i)++;
      updated = true;
    } else {
      state.at(i) = range.at(0) + 1;
    }
  }

  if (updated) {

    int negsum = -1 * std::accumulate(begin(state), end(state) - 1, 0);

    if ((range.at(0) < negsum) && (negsum < range.at(1))) {
      state.at(n - 1) = negsum;
    } else {
      updated = update_state(state, range, n);
    }
  }

  return updated;
}

void Walk::fill_matrices(const Eigen::VectorXd &p_step) {

  std::vector<int> state;
  state.resize(n_alternatives);
  std::fill(begin(state), end(state), 0);

  Q.setIdentity();
  Q = Q * (1 - p_step.sum());
  R.setZero();
  Z.setZero();
  Z(0, transient_state_index.at(state)) = 1;

  for (const auto &prev : transient_state_index) {
    // fill in transition probability from prev

    for (unsigned int alt = 0; alt < n_alternatives; alt++) {

      // relative evidence for the state from which we transition
      state = prev.first;
      // alt accumulates relative evidence by n_alternatives - 1.
      // 1 is subtracted within a for loop below.
      state.at(alt) += n_alternatives;
      // other alternatives loses relative evidence by 1
      for (unsigned int oth = 0; oth < n_alternatives; oth++) {
        state.at(oth) += -1;
      }

      if (state.at(alt) < boundary) {
        // transition within transient states
        Q(prev.second, transient_state_index.at(state)) = p_step(alt);
      } else if (state.at(alt) >= boundary) {
        // transition to an aborbing state
        R(prev.second, alt) = p_step(alt);
      }
    }
  }

  // check if rows sum to 1
  Eigen::MatrixXd rowSums = Q.rowwise().sum() + R.rowwise().sum();
  for (unsigned int i = 0; i < n_transient_states; i++) {
    if (fabs(rowSums(i) - 1) > 1e-10) {
      throw std::runtime_error("A row in transition matrix has to sum to 1.");
    }
  }
}
