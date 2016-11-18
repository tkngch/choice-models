// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include <eigen3/Eigen/Core>
#include "mdft.hpp"

void test_predicting_effects(MDFT::Model &model,
                             const MDFT::Parameter &parameter);
int test_predicting_effect(MDFT::Model &model, const MDFT::Parameter &parameter,
                           const Eigen::MatrixXd &M,
                           const Eigen::VectorXd &correct);

void test_simulating_effects(MDFT::Model &model, MDFT::Parameter &parameter);
int test_simulating_effect(MDFT::Model &model, const MDFT::Parameter &parameter,
                           const Eigen::MatrixXd &M,
                           const Eigen::VectorXd &correct);

int assess_result(const Eigen::VectorXd &correct,
                  const std::vector<double> &computed,
                  const double tolerance = 0.001);
