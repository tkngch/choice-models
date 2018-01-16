// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include "dbs.hpp"
#include <eigen3/Eigen/Core>
#include <random>

void random_set_parameter(DbS::Parameter &parameter);

void test_computation(DbS::Model &model, DbS::Parameter &parameter);
void test_computation_with_memory_samples(DbS::Model &model,
                                          const DbS::Parameter &parameter);
void test_computation_with_missing_values(DbS::Model &model,
                                          const DbS::Parameter &parameter);
void test_computation_with_missing_values2(DbS::Model &model,
                                           const DbS::Parameter &parameter);

void test_predicting_effects(DbS::Model &model, DbS::Parameter &parameter);

void test_predicting_effect(DbS::Model &model, const DbS::Parameter &parameter,
                            const Eigen::MatrixXd &choice_set,
                            const std::vector<Eigen::VectorXd> &memory_set,
                            const Eigen::VectorXd &correct);

void test_predicting_effect(DbS::Model &model, const DbS::Parameter &parameter,
                            const Eigen::MatrixXd &choice_set,
                            const Eigen::VectorXd &correct);

void assess_result(const Eigen::VectorXd &correct,
                   const std::vector<double> &computed,
                   const double tolerance = 1e-10);
