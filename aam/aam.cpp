// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include "aam.hpp"

namespace AAM {

void Parameter::print(void) { std::cout << get_string() << std::endl; };

std::string Parameter::get_string(void) {
  std::stringstream ss;
  ss << "\"[" << a0 << ", " << alpha << ", " << e << ", " << d << "]\"";
  return ss.str();
};

void Model::predict(std::vector<double> &choice_probability,
                    const Eigen::MatrixXd &M, const Parameter &parameter) {

  n_alternatives = M.rows();
  n_dimensions = M.cols();

  choice_probability.resize(n_alternatives);
  std::fill(begin(choice_probability), end(choice_probability), 0.0);

  Eigen::MatrixXd S(n_alternatives, n_alternatives);
  Eigen::MatrixXd V(n_dimensions, n_alternatives);
  Eigen::VectorXd W(n_dimensions);

  construct_feedback_matrix(S, parameter);
  compute_attribute_accessibility(W, M, parameter);
  compute_attribute_values(V, M, parameter);

  std::random_device rd;
  std::mt19937 rng(rd());

  int winner;
  unsigned int n = 0;
  for (unsigned int i = 0; i < n_simulations; i++) {
    winner = simulate(rng, S, W, V, parameter);
    if (winner >= 0) {
      choice_probability.at(winner) += 1.0;
      n++;
    }
  }

  if (n > 0) {
    std::transform(begin(choice_probability), end(choice_probability),
                   begin(choice_probability),
                   [&n](const double x) { return x / n; });
  }
}

void Model::compute(std::vector<double> &expected_preference,
                    const Eigen::MatrixXd &M, const Parameter &parameter) {

  n_alternatives = M.rows();
  n_dimensions = M.cols();

  Eigen::MatrixXd S(n_alternatives, n_alternatives);
  Eigen::MatrixXd V(n_dimensions, n_alternatives);
  Eigen::VectorXd W(n_dimensions);
  Eigen::VectorXd U(n_alternatives);

  construct_feedback_matrix(S, parameter);
  compute_attribute_accessibility(W, M, parameter);
  compute_attribute_values(V, M, parameter);

  U = V * W;

  expected_preference.resize(n_alternatives);
  for (unsigned int i = 0; i < n_alternatives; i++) {
    expected_preference.at(i) = U(i);
  }
}

void Model::construct_feedback_matrix(Eigen::MatrixXd &S,
                                      const Parameter &parameter) {
  assert(parameter.d >= 0);

  S.setZero();
  for (unsigned int i = 0; i < n_alternatives; i++) {
    S(i, i) = parameter.d;
  }
}

void Model::compute_attribute_accessibility(Eigen::VectorXd &W,
                                            const Eigen::MatrixXd &M,
                                            const Parameter &parameter) {

  W.resize(n_dimensions);
  W.setZero();

  for (unsigned int i = 0; i < n_dimensions; i++) {
    for (unsigned int j = 0; j < n_alternatives; j++) {
      W(i) += fabs(M(j, i));
    }
    W(i) += parameter.a0;
  }

  double denom = W.sum();
  W /= denom;
}

void Model::compute_attribute_values(Eigen::MatrixXd &V,
                                     const Eigen::MatrixXd &M,
                                     const Parameter &parameter) {

  V.resize(n_alternatives, n_dimensions);
  for (unsigned int i = 0; i < n_dimensions; i++) {
    for (unsigned int j = 0; j < n_alternatives; j++) {

      if (M(j, i) > 0) {
        V(j, i) = pow(M(j, i), parameter.alpha);
      } else if (M(j, i) < 0) {
        V(j, i) = -1 * pow(fabs(M(j, i)), parameter.alpha);
      } else {
        V(j, i) = 0;
      }
    }
  }
}

int Model::simulate(std::mt19937 &rng, const Eigen::MatrixXd &S,
                    const Eigen::MatrixXd &W, const Eigen::MatrixXd &V,
                    const Parameter &parameter) {

  std::uniform_real_distribution<double> runif(0, 1);
  std::normal_distribution<double> rnorm(0.0, parameter.e);

  Eigen::MatrixXd P(n_alternatives, 1);
  P.setZero();

  double u;
  unsigned int dim;
  Eigen::MatrixXd noise(n_alternatives, 1);

  for (unsigned int iter = 0; iter < n_iterations; iter++) {

    u = runif(rng);
    dim = n_dimensions;
    for (unsigned int i = 0; i < n_dimensions; i++) {
      if (u < W(i)) {
        dim = i;
        u = 100;
      } else {
        u -= W(i);
      }
    }

    for (unsigned int i = 0; i < n_alternatives; i++) {
      noise(i, 0) = rnorm(rng);
    }

    P = S * P + V.col(dim) + noise;
  }

  int winner = find_winner(P);
  return winner;
}

int Model::find_winner(Eigen::MatrixXd &P) {

  Eigen::MatrixXd::Index row, col;
  P.maxCoeff(&row, &col);

  std::vector<unsigned int> winners;
  const double practical_zero = 1e-10;
  for (unsigned int i = 0; i < n_alternatives; i++) {
    if ((fabs(P(row, 0) - P(i, 0)) < practical_zero) && (P(row, 0) > 0) &&
        (P(i, 0) > 0)) {
      winners.push_back(i);
    }
  }

  int winner;
  if (winners.size() > 0) {
    std::random_shuffle(begin(winners), end(winners));
    winner = winners.at(0);
  } else {
    winner = -1;
  }

  return winner;
}

}; // namespace AAM
