// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include "mlba.hpp"

namespace MLBA {

void Parameter::print(void) { std::cout << get_string() << std::endl; };

std::string Parameter::get_string(void) {
  std::stringstream ss;
  ss << "\"[" << i0 << ", " << m << ", " << lambda1 << ", " << lambda2 << "]\"";
  return ss.str();
};

void Model::predict(std::vector<double> &choice_probability,
                    const Eigen::MatrixXd &alternatives,
                    const Parameter &parameter) {

  std::vector<double> drift_rate;
  compute(drift_rate, alternatives, parameter);
  Eigen::VectorXd drift =
      Eigen::VectorXd::Map(drift_rate.data(), drift_rate.size());

  choice_probability.resize(n_alternatives);
  if (drift.maxCoeff() <= 0) {
    // random choice
    std::fill(begin(choice_probability), end(choice_probability), 1.0 / n_alternatives);
    // throw std::runtime_error("Drift rates are all negative.");
    // std::cout << "Drift rates are all negative." << std::endl;
    return;
  }

  Eigen::MatrixXd PermMat(n_alternatives, n_alternatives);
  Eigen::VectorXd d;

  for (unsigned int i = 0; i < n_alternatives; i++) {

    PermMat.setIdentity();
    PermMat(i, i) = 0;
    PermMat(0, i) = 1;
    for (unsigned int j = 0; j < i; j++) {
      PermMat(j, j) = 0;
      PermMat(j + 1, j) = 1;
    }
    d = PermMat * drift;

    choice_probability.at(i) = n1CDF(d);
  }
}

// void Model::normalise_choice_probability(
//     std::vector<double> &choice_probability) {
//
//   unsigned int n_alternatives = choice_probability.size();
//   double min =
//       *std::min_element(begin(choice_probability), end(choice_probability));
//
//   double zero = 1e-8;
//
//   if (min < zero) {
//     std::transform(begin(choice_probability), end(choice_probability),
//                    begin(choice_probability),
//                    bind2nd(std::minus<double>(), min));
//     // [&min](double x) { return x - min; });
//   }
//
//   for (unsigned int i = 0; i < n_alternatives; i++) {
//     // look for nan
//     if (!(choice_probability.at(i) > 0)) {
//       choice_probability.at(i) = 0;
//     }
//   }
//
//   double sum =
//       std::accumulate(begin(choice_probability), end(choice_probability), 0.0);
//   if (sum <= zero) {
//     std::fill(begin(choice_probability), end(choice_probability),
//               1.0 / n_alternatives);
//   } else if (fabs(sum - 1.0) > zero) {
//     std::transform(begin(choice_probability), end(choice_probability),
//                    begin(choice_probability),
//                    std::bind1st(std::multiplies<double>(), 1.0 / sum));
//     // [&sum](double x) { return x / sum; });
//   }
// }

double Model::n1CDF(Eigen::VectorXd &d) {

  gsl_function F;
  F.function = &n1PDF;
  F.params = &d;

  double result, error;

  // integration sometimes fails
  gsl_set_error_handler_off();

  int limit = 10000;
  gsl_integration_workspace *w = gsl_integration_workspace_alloc(limit);

  double epsabs = 0, epsrel = 1e-7;
  double lower = 0, upper;

  const double chi = 2,
               sdI = 1; // following assumes these values. do not change these.

  int status = gsl_integration_qagiu(&F, lower, epsabs, epsrel, limit, w,
                                     &result, &error);

  if (status != GSL_SUCCESS) {

    double mean_drift = d.mean();
    if (mean_drift > d(0)) {
      lower = (chi - 0.98) / (mean_drift + 2 * sdI);
    } else {
      lower = (chi - 0.98) / (d(0) + 2 * sdI);
    }

    upper = 0.02 * chi / (mean_drift - 2 * sdI);

    size_t neval;
    status = gsl_integration_qng(&F, lower, upper, epsabs, epsrel, &result,
                                 &error, &neval);
  }

  if (status != GSL_SUCCESS) {
    // std::cout << "Integration Failed" << std::endl;
    // result = GSL_NAN;
    result = 0.0;
  }

  gsl_integration_workspace_free(w);

  return result;
}

double Model::n1PDF(double x, void *params) {

  Eigen::VectorXd d = *(Eigen::VectorXd *)params;

  if (fabs(x) < 1e-10) {
    return 0;
  }

  // fptpdf and fptcdf assume these parameter values (i.e., x0max = 1, chi = 2,
  // sdI = 1), so don't change them.
  const double x0max = 1;
  const double chi = 2;
  const double sdI = 1;

  double res = fptpdf(x, x0max, chi, d(0), sdI);

  for (unsigned int i = 1; i < d.size(); i++) {
    res *= 1 - fptcdf(x, x0max, chi, d(i), sdI);
  }

  double normaliser = 1;
  for (unsigned int i = 0; i < d.size(); i++) {
    normaliser *= gsl_cdf_gaussian_P(-d(i), sdI);
  }

  return res / (1 - normaliser);
}

double Model::fptcdf(const double &z, const double &x0max, const double &chi,
                     const double &d, const double &sddrift) {

  const double zs = z * sddrift;
  const double zu = z * d;

  const double chiminuszu = chi - zu;
  const double xx = chiminuszu - x0max;
  const double chizu = chiminuszu / zs;
  const double chizumax = xx / zs;

  const double tmp1 =
      zs * (gsl_ran_gaussian_pdf(chizumax, 1) - gsl_ran_gaussian_pdf(chizu, 1));
  const double tmp2 = xx * gsl_cdf_gaussian_P(chizumax, 1) -
                      chiminuszu * gsl_cdf_gaussian_P(chizu, 1);
  double res = 1 + (tmp1 + tmp2) / x0max;

  return res;
}

double Model::fptpdf(const double &z, const double &x0max, const double &chi,
                     const double &d, const double &sddrift) {

  const double zs = z * sddrift;
  const double zu = z * d;

  const double chiminuszu = chi - zu;
  const double chizu = chiminuszu / zs;
  const double chizumax = (chiminuszu - x0max) / zs;

  const double tmp1 =
      d * (gsl_cdf_gaussian_P(chizu, 1) - gsl_cdf_gaussian_P(chizumax, 1));
  const double tmp2 = sddrift * (gsl_ran_gaussian_pdf(chizumax, 1) -
                                 gsl_ran_gaussian_pdf(chizu, 1));
  double res = (tmp1 + tmp2) / x0max;
  return res;
}

void Model::compute(std::vector<double> &drift_rate, const Eigen::MatrixXd &M,
                    const Parameter &parameter) {

  n_alternatives = M.rows();
  n_dimensions = M.cols();

  drift_rate.resize(n_alternatives);
  std::fill(begin(drift_rate), end(drift_rate), 0.0);

  std::vector<std::vector<double>> U;
  compute_subjective_value(U, M, parameter);

  drift_rate.resize(n_alternatives);
  for (unsigned int i = 0; i < n_alternatives; i++) {
    drift_rate.at(i) = compute_drift_rate(i, U, parameter);
  }
}

void Model::compute_subjective_value(std::vector<std::vector<double>> &U,
                                     const Eigen::MatrixXd &M,
                                     const Parameter &parameter) {

  assert(n_dimensions == 2);
  U.resize(n_alternatives);

  double a, b, angle;

  for (unsigned int i = 0; i < n_alternatives; i++) {

    U.at(i).resize(n_dimensions);

    a = M.row(i).sum();
    b = a;

    angle = atan(M.row(i)(1) / M.row(i)(0));

    U.at(i).at(0) =
        b / pow(pow(tan(angle), parameter.m) + pow(b / a, parameter.m),
                1 / parameter.m);

    U.at(i).at(1) =
        b * pow(1 - pow(U.at(i).at(0) / a, parameter.m), 1 / parameter.m);
  }
}

double Model::compute_drift_rate(const unsigned int i,
                                 const std::vector<std::vector<double>> &U,
                                 const Parameter &parameter) {

  auto compute_weight = [&parameter](const double a, const double b) {

    double wt;
    if (a > b) {
      wt = exp(-1 * parameter.lambda1 * fabs(a - b));
    } else {
      wt = exp(-1 * parameter.lambda2 * fabs(a - b));
    }

    return wt;
  };

  auto evaluate = [&](const unsigned int j) {

    double v = 0;
    for (unsigned int dim = 0; dim < n_dimensions; dim++) {
      v += compute_weight(U.at(i).at(dim), U.at(j).at(dim)) *
           (U.at(i).at(dim) - U.at(j).at(dim));
    }

    return v;
  };

  double d = 0;
  for (unsigned int j = 0; j < n_alternatives; j++) {
    if (i != j) {
      d += evaluate(j);
    }
  }
  d += parameter.i0;

  return d;
}

}; // namespace MDFT
