// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include "mdft.hpp"

namespace MDFT {

void Parameter::print(void) { std::cout << get_string() << std::endl; };

std::string Parameter::get_string(void) {
  std::stringstream ss;
  ss << "\"[" << wgt << ", " << phi1 << ", " << phi2 << ", " << sig2 << ", "
     << stopping_time << "]\"";
  return ss.str();
};

void Model::predict(std::vector<double> &choice_probability,
                    const Eigen::MatrixXd &M, const Parameter &parameter) {

  if (parameter.stopping_time < 0) {
    std::runtime_error("Stopping Time has to be non-negative.");
  }

  n_alternatives = M.rows();
  n_dimensions = M.cols();

  choice_probability.resize(n_alternatives);
  std::fill(begin(choice_probability), end(choice_probability), 0.0);

  Eigen::MatrixXd S(n_alternatives, n_alternatives);
  construct_feedback_matrix(S, M, parameter);
  // unsigned int res = check_feedback_matrix(S);
  //
  // if (res == 0) {

  Eigen::MatrixXd C(n_alternatives, n_alternatives);
  construct_contrast_matrix(C);

  Eigen::VectorXd W(n_dimensions);
  construct_weight_vector(W);

  Eigen::VectorXd Eta(n_alternatives);
  Eigen::MatrixXd Omega(n_alternatives, n_alternatives);
  construct_preference_and_covariance(Eta, Omega, C, M, W, S, parameter);

  Eigen::MatrixXd L(n_alternatives - 1, n_alternatives);
  for (unsigned int i = 0; i < n_alternatives; i++) {
    construct_L_matrix(L, i);
    choice_probability.at(i) = compute_choice_probability(L, Eta, Omega);
  }

  // }
}

void Model::simulate(std::vector<double> &choice_probability,
                     const Eigen::MatrixXd &M, const Parameter &parameter,
                     const unsigned int n_simulations,
                     const unsigned int max_iter) {

  if (parameter.stopping_time < 0) {
    std::runtime_error("Stopping Time has to be non-negative.");
  }

  n_alternatives = M.rows();
  n_dimensions = M.cols();

  choice_probability.resize(n_alternatives);
  std::fill(begin(choice_probability), end(choice_probability), 0.0);

  Eigen::MatrixXd S(n_alternatives, n_alternatives);
  Eigen::MatrixXd C(n_alternatives, n_alternatives);
  Eigen::MatrixXd W(n_dimensions, n_dimensions);

  construct_feedback_matrix(S, M, parameter);
  unsigned int res = check_feedback_matrix(S);

  if (res == 0) {

    construct_contrast_matrix(C);
    W.setIdentity();

    std::random_device rd;
    std::mt19937 rng(rd());

    int winner;
    unsigned int n = 0;
    for (unsigned int i = 0; i < n_simulations; i++) {
      winner = run(rng, S, C, M, W, parameter, max_iter);
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
}

int Model::run(std::mt19937 &rng, const Eigen::MatrixXd &S,
               const Eigen::MatrixXd &C, const Eigen::MatrixXd &M,
               const Eigen::MatrixXd &W, const Parameter &parameter,
               const unsigned int max_iter) {

  std::uniform_int_distribution<int> runif(0, n_dimensions - 1);
  std::normal_distribution<double> rnorm(0.0, 1.0);

  Eigen::MatrixXd V(n_alternatives, 1);
  V.setZero();
  Eigen::MatrixXd P(n_alternatives, 1);
  P.setZero();

  unsigned int dim;
  unsigned int iter = 0;
  Eigen::MatrixXd noise(n_alternatives, 1);

  do {
    dim = runif(rng);

    for (unsigned int i = 0; i < n_alternatives; i++) {
      noise(i, 0) = rnorm(rng);
    }

    V = C * M * W.col(dim) + parameter.sig2 * C * noise;
    P = S * P + V;

    iter++;

  } while ((P.maxCoeff() < parameter.theta) && (iter < max_iter));

  int winner;

  if (iter < max_iter) {
    Eigen::MatrixXd::Index row, col;
    P.maxCoeff(&row, &col);
    winner = row;
  } else {
    winner = -1;
  }

  return winner;
}

void Model::construct_feedback_matrix(Eigen::MatrixXd &S,
                                      const Eigen::MatrixXd &M,
                                      const Parameter &parameter) {
  S.setZero();

  Eigen::Matrix2d T;
  T.setConstant(1.0 / sqrt(2));
  T(0, 0) = -1 * T(0, 0);

  Eigen::DiagonalMatrix<double, 2> W(1, parameter.wgt);

  Eigen::MatrixXd DV(n_dimensions, 1);

  double s;
  for (unsigned int i = 0; i < n_alternatives; i++) {
    for (unsigned int j = 0; j < n_alternatives; j++) {

      for (unsigned int k = 0; k < n_dimensions; k++) {
        DV(k, 0) = M(i, k) - M(j, k);
      }
      DV = T * DV;
      s = (DV.transpose() * W * DV)(0, 0);
      S(i, j) = parameter.phi2 * exp(-1 * parameter.phi1 * s * s);
    }
  }

  S = Eigen::MatrixXd::Identity(n_alternatives, n_alternatives) - S;
}

unsigned int Model::check_feedback_matrix(Eigen::MatrixXd &S) {

  Eigen::VectorXcd eigenvalues = S.eigenvalues();
  Eigen::VectorXd real = eigenvalues.real();
  Eigen::VectorXd imag = eigenvalues.imag();

  double magnitude, max_magnitude = 0;
  for (unsigned int i = 0; i < n_alternatives; i++) {
    magnitude = sqrt(real(i) * real(i) + imag(i) * imag(i));
    if (magnitude > max_magnitude) {
      max_magnitude = magnitude;
    }
  }

  // unsigned int res = 1;
  unsigned int res = 0;
  if (max_magnitude < 1) {
    res = 0;
  }

  return res;
}

void Model::construct_contrast_matrix(Eigen::MatrixXd &C) {

  if (n_alternatives > 1) {
    C.setConstant(-1.0 / (n_alternatives - 1.0));
  } else {
    C.setZero();
  }

  for (unsigned int i = 0; i < n_alternatives; i++) {
    C(i, i) = 1.0;
  }
}

void Model::construct_weight_vector(Eigen::VectorXd &W) {

  W.setConstant(1.0 / n_dimensions);
}

void Model::construct_preference_and_covariance(
    Eigen::VectorXd &Eta, Eigen::MatrixXd &Omega, const Eigen::MatrixXd &C,
    const Eigen::MatrixXd &M, const Eigen::VectorXd &W,
    const Eigen::MatrixXd &S, const Parameter &parameter) {

  Eigen::VectorXd Mu = C * M * W;
  Eta = Mu;

  Eigen::MatrixXd Psi = W.asDiagonal();
  Psi -= (W * W.transpose());

  Eigen::MatrixXd I(n_alternatives, n_alternatives);
  I.setIdentity();
  Eigen::MatrixXd Phi = C * M * Psi * M.transpose() * C.transpose() +
                        parameter.sig2 * C * I * C.transpose();

  Omega = Phi;
  double rt;
  Eigen::MatrixXd Si = I;

  for (int i = 2; i <= parameter.stopping_time; i++) {
    rt = 1.0 / (1 + exp(i - 202.0) / 25); // [Takao] Not sure what this is.
    Si = S * Si;
    Eta += rt * Si * Mu;
    Omega += pow(rt, 2) * Si * Phi * Si.transpose();
  }
}

void Model::construct_L_matrix(Eigen::MatrixXd &L, const unsigned int i) {

  L.setZero();
  for (unsigned int row = 0; row < n_alternatives - 1; row++) {
    for (unsigned int col = 0; col < n_alternatives; col++) {
      if (col == i) {
        L(row, col) = 1;
      } else if ((col < i) && (row == col)) {
        L(row, col) = -1;
      } else if ((col > i) && (row == col - 1)) {
        L(row, col) = -1;
      }
    }
  }
}

double Model::compute_choice_probability(const Eigen::MatrixXd &L,
                                         const Eigen::MatrixXd &Eta,
                                         const Eigen::MatrixXd &Omega) {

  Eigen::VectorXd Leta = L * Eta;
  Eigen::MatrixXd Lomega = L * Omega * L.transpose();

  double p = 0;
  // if (Lomega.determinant() >= 0) {
  // this determinant computation is not reliable: i.e., same input can lead to
  // different output.
  p = multivariate_normal_cdf(Leta, Lomega);
  // }

  return p;
}

double Model::multivariate_normal_cdf(const Eigen::VectorXd &Leta,
                                      const Eigen::MatrixXd &Lomega) {

  unsigned int n = Leta.size();

  std::vector<double> Is;
  Is.resize(n);
  double lower[n];

  for (unsigned int i = 0; i < n; i++) {
    Is.at(i) = sqrt(1 / Lomega(i, i));
    lower[i] = -1 * Leta(i) / sqrt(Lomega(i, i));
  }

  unsigned int n_corr_elements = (n * (n - 1)) / 2;
  double corr[n_corr_elements];
  unsigned int i = 0;
  for (unsigned int col = 0; col < n; col++) {
    for (unsigned int row = 0; row < col; row++) {
      corr[i++] = Is.at(row) * Lomega(row, col) * Is.at(col);
    }
  }
  // assert(i == n_corr_elements);

  double err;
  double cdf = pmvnorm_Q(n_alternatives - 1, lower, corr, &err);
  if (err > 1e-4) {
    cdf = nan("");
  }
  return cdf;
}

}; // namespace MDFT
