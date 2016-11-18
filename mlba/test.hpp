// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

/* This script computes drift rates and compare against what are computed with
 * Trueblood's script.
 */

#include <iostream>
#include <eigen3/Eigen/Core>
#include "mlba.hpp"

void test_predicting_effects(MLBA::Model &model, MLBA::Parameter &parameter);

// test with non standard parameter values
void test_predicting_effects2(MLBA::Model &model, MLBA::Parameter &parameter);

int test_predicting_effect(MLBA::Model &model, const MLBA::Parameter &parameter,
                           Eigen::MatrixXd &M,
                           const Eigen::VectorXd &correct_rate);

void test_effects(MLBA::Model &model, const MLBA::Parameter &parameter);
int test_effect(MLBA::Model &model, const MLBA::Parameter &parameter,
                Eigen::MatrixXd &M, const Eigen::VectorXd &correct_rate);

int assess_results(const Eigen::VectorXd &correct_rate,
                   const std::vector<double> &drift_rate);
