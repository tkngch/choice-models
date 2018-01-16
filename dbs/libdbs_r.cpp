/*
 * Filename: libdbs_r.cpp

 * R interface for decision by sampling model.

 * Copyright (C) 2015, 2017 Takao Noguchi (tkngch@runbox.com)
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.

 */

#include <Rcpp.h>
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "dbs.hpp"

DbS::Parameter set_parameter(SEXP &parameter_) {

  std::vector<double> param_vec = Rcpp::as<std::vector<double>>(parameter_);
  assert(param_vec.size() == 4);
  DbS::Parameter parameter{param_vec.at(0), param_vec.at(1), param_vec.at(2),
                           param_vec.at(3)};
  return parameter;
}

Eigen::MatrixXd
set_choiceset(const std::vector<std::vector<double>> choiceset) {

  const unsigned int n_dimensions = choiceset.at(0).size();
  const unsigned int n_alternatives = choiceset.size();
  Eigen::MatrixXd M(n_alternatives, n_dimensions);

  for (unsigned int i = 0; i < n_alternatives; i++) {
    M.row(i) = Eigen::VectorXd::Map(&choiceset.at(i).at(0), n_dimensions);
  }

  return M;
}

std::vector<Eigen::VectorXd>
set_memoryset(const std::vector<std::vector<double>> memory_sample) {

  const unsigned int n_samples = memory_sample.size();
  std::vector<Eigen::VectorXd> M;
  M.resize(n_samples);

  for (unsigned int i = 0; i < n_samples; i++) {
    M.at(i) = Eigen::VectorXd::Map(&memory_sample.at(i).at(0),
                                   memory_sample.at(i).size());
  }

  return M;
}

RcppExport SEXP evaluate(SEXP choiceset_, SEXP parameter_) {
  DbS::Model model;
  DbS::Parameter parameter = set_parameter(parameter_);

  const Eigen::MatrixXd choiceset =
      set_choiceset(Rcpp::as<std::vector<std::vector<double>>>(choiceset_));

  std::vector<double> choice_probability;
  model.predict(choice_probability, choiceset, parameter);

  return Rcpp::wrap(choice_probability);
}

RcppExport SEXP evaluate_number_of_comparisons(SEXP choiceset_,
                                               SEXP parameter_) {
  DbS::Model model;
  DbS::Parameter parameter = set_parameter(parameter_);

  const Eigen::MatrixXd choiceset =
      set_choiceset(Rcpp::as<std::vector<std::vector<double>>>(choiceset_));

  std::vector<double> choice_probability;
  double expected_number_of_comparisons;
  model.predict(choice_probability, expected_number_of_comparisons, choiceset,
                parameter);

  return Rcpp::List::create(Rcpp::Named("choice_probability") =
                                Rcpp::wrap(choice_probability),
                            Rcpp::Named("expected_number_of_comparisons") =
                                Rcpp::wrap(expected_number_of_comparisons));
}

RcppExport SEXP evaluate_with_memory_sample(SEXP choiceset_,
                                            SEXP memory_sample_,
                                            SEXP parameter_) {
  DbS::Model model;
  DbS::Parameter parameter = set_parameter(parameter_);

  const Eigen::MatrixXd choiceset =
      set_choiceset(Rcpp::as<std::vector<std::vector<double>>>(choiceset_));

  const std::vector<Eigen::VectorXd> memory_sample =
      set_memoryset(Rcpp::as<std::vector<std::vector<double>>>(memory_sample_));

  std::vector<double> choice_probability;
  model.predict(choice_probability, choiceset, memory_sample, parameter);

  return Rcpp::wrap(choice_probability);
}
