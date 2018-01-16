// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

/* This script compares outputs from my code against output from
 * Bathia's script from the paper's supplement matrial.
 */

#include <iostream>
#include <eigen3/Eigen/Core>
#include "aam.hpp"

void test_effects(AAM::Model &model, AAM::Parameter &parameter);
int test_effect(AAM::Model &model, const AAM::Parameter &parameter,
                const Eigen::MatrixXd &M, const Eigen::VectorXd &correct_rate);

int assess_result(const Eigen::VectorXd &correct,
                  const std::vector<double> &computed);
