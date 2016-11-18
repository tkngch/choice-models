// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include <cmath>
#include <iostream>
#include <vector>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_randist.h>
#include <eigen3/Eigen/Dense>

namespace MLBA {

class Parameter {
public:
  Parameter(void){};
  Parameter(double m, double lambda1, double lambda2, double i0)
      : m(m), lambda1(lambda1), lambda2(lambda2), i0(i0){};

  ~Parameter(){};

  void print(void);
  std::string get_string(void);

  double m, lambda1, lambda2, i0;

}; // class Parameter

class Model {

public:
  Model(void){};
  ~Model(){};

  // This function just returns drift rate, not choice probabilities.
  void compute(std::vector<double> &drift_rate,
               const Eigen::MatrixXd &alternatives, const Parameter &parameter);

  void predict(std::vector<double> &choice_probability,
               const Eigen::MatrixXd &alternatives, const Parameter &parameter);

private:
  unsigned int n_alternatives, n_dimensions;

  // This function assumes that n_dimensions == 2;
  void compute_subjective_value(std::vector<std::vector<double>> &U,
                                const Eigen::MatrixXd &M,
                                const Parameter &parameter);

  double compute_drift_rate(const unsigned int index,
                            const std::vector<std::vector<double>> &U,
                            const Parameter &parameter);

  // The following four methods (n1CDF, n1PDF, fptpdf, and fptcdf) are direct
  // translation of Trueblood's Matlab functions.
  double n1CDF(Eigen::VectorXd &d);
  static double n1PDF(double x, void *params);

  static double fptpdf(const double &z, const double &x0max, const double &chi,
                       const double &d, const double &sddrift);
  static double fptcdf(const double &z, const double &x0max, const double &chi,
                       const double &d, const double &sddrift);

}; // Model

}; // namespace MDFT
