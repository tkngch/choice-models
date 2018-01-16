// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include "test.hpp"

int main(void) {

  DbS::Model model;
  DbS::Parameter parameter;

  test_predicting_effects(model, parameter);
  // test_computation(model, parameter);
}

void random_set_parameter(DbS::Parameter &parameter) {
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<> dist_similarity_slope(2, 5);
  std::uniform_real_distribution<> dist_recognition_intercept(0.01, 0.2);
  std::uniform_real_distribution<> dist_recognition_slope(-100, -1);
  std::uniform_real_distribution<> dist_threshold(0.01, 3);

  parameter.similarity_slope = dist_similarity_slope(gen);
  parameter.recognition_intercept = dist_recognition_intercept(gen);
  parameter.recognition_slope = dist_recognition_slope(gen);
  parameter.threshold = dist_threshold(gen);

  // parameter.similarity_slope = 3.0;
  // parameter.recognition_intercept = 0.1;
  // parameter.recognition_slope = -50;
  // parameter.threshold = 1.0;
}

void random_generate_choiceset(Eigen::MatrixXd &choice_set,
                               const unsigned int n_alternatives) {

  const unsigned int n_dimensions = 2;

  std::random_device rd;
  std::mt19937 gen(rd());

  std::normal_distribution<> dist_attribute_value1(100, 100); // mean and sd
  std::normal_distribution<> dist_attribute_value2(0, 10);

  choice_set.resize(n_alternatives, n_dimensions);
  choice_set.setOnes();

  for (unsigned int i = 0; i < n_alternatives; i++) {
    choice_set(i, 0) = dist_attribute_value1(gen);
    choice_set(i, 1) = -1 * choice_set(i, 0) + dist_attribute_value1(gen);
  }
}

void test_computation(DbS::Model &model, DbS::Parameter &parameter) {

  const unsigned int n_tests = 20;
  typedef void (*functioncall)(DbS::Model & model,
                               const DbS::Parameter &parameter);

  auto test_comp = [&model, &parameter, &n_tests](const std::string head,
                                                  functioncall function) {
    std::cout << head << ": ";
    for (unsigned int i = 0; i < n_tests; i++) {
      random_set_parameter(parameter);
      function(model, parameter);
    }
    std::cout << " Pass" << std::endl;
  };

  test_comp("Testing with memory samples",
            test_computation_with_memory_samples);

  test_comp("Testing with missing values",
            test_computation_with_missing_values);

  test_comp("Testing with missing values 2",
            test_computation_with_missing_values2);
}

void test_computation_with_memory_samples(DbS::Model &model,
                                          const DbS::Parameter &parameter) {

  // p(A from {A, B, C}) / [p(A from {A, B, C}) + p(A from {A, B, C})]
  // should be equal to p(A from {A, B}) when C is in memory set.

  // NOTE: Actually, it should not. With more alternatives in a choice set, an
  // alternative does not lose as much relative evidence when another
  // alternative wins a comparison. As a result, it will be quicker to reach a
  // choice, and more alternatives are more likely to win with more
  // alternatives.

  unsigned int n_alternatives = 3;
  unsigned int n_dimensions = 2;

  Eigen::VectorXd correct(n_alternatives - 1);

  Eigen::MatrixXd choice_set1;
  random_generate_choiceset(choice_set1, n_alternatives);

  Eigen::MatrixXd choice_set2(n_alternatives - 1, n_dimensions);
  for (unsigned int i = 0; i < n_alternatives - 1; i++) {
    for (unsigned int j = 0; j < n_dimensions; j++) {
      choice_set2(i, j) = choice_set1(i, j);
    }
  }

  std::vector<Eigen::VectorXd> memory_set;
  memory_set.resize(n_dimensions);
  for (unsigned int j = 0; j < n_dimensions; j++) {
    memory_set.at(j).resize(1);
    memory_set.at(j) << choice_set1(n_alternatives - 1, j);
  }

  std::vector<double> choice_probability1;
  std::vector<double> choice_probability2;

  model.predict(choice_probability1, choice_set1, parameter);
  model.predict(choice_probability2, choice_set2, memory_set, parameter);

  const double normaliser =
      choice_probability1.at(0) + choice_probability1.at(1);

  correct << choice_probability1.at(0) / normaliser,
      choice_probability1.at(1) / normaliser;
  assess_result(correct, choice_probability2);
}

void test_computation_with_missing_values(DbS::Model &model,
                                          const DbS::Parameter &parameter) {

  // adding an alternative with all missing values should not alter choice
  // probability.
  //
  // NOTE: Actually, it does. With more alternatives in a choice set, an
  // alternative does not lose as much relative evidence when another
  // alternative wins a comparison. As a result, it will be quicker to reach a
  // choice, and more alternatives are more likely to win with more
  // alternatives.

  unsigned int n_alternatives = 2;
  unsigned int n_dimensions = 2;

  Eigen::VectorXd correct(n_alternatives);
  correct.setOnes();

  Eigen::MatrixXd choice_set1;
  random_generate_choiceset(choice_set1, n_alternatives);

  Eigen::MatrixXd choice_set2(n_alternatives + 1, n_dimensions);
  for (unsigned int j = 0; j < n_dimensions; j++) {
    for (unsigned int i = 0; i < n_alternatives; i++) {
      choice_set2(i, j) = choice_set1(i, j);
    }
    choice_set2(n_alternatives, j) = std::numeric_limits<double>::quiet_NaN();
  }

  std::vector<double> choice_probability1;
  std::vector<double> choice_probability2;

  model.predict(choice_probability1, choice_set1, parameter);
  model.predict(choice_probability2, choice_set2, parameter);

  for (unsigned int i = 0; i < n_alternatives; i++) {
    correct(i) = choice_probability2.at(i);
  }
  std::cout << correct << std::endl;
  assess_result(correct, choice_probability1);
}

void test_computation_with_missing_values2(DbS::Model &model,
                                           const DbS::Parameter &parameter) {

  // probability of A for [[Ax, Ay], [Bx, By]] should be the same sa the
  // probability of A for [[Ax, Ay], [Bx, NA], [NA, By]].

  // NOTE: Actually, it should not. With more alternatives in a choice set, an
  // alternative does not lose as much relative evidence when another
  // alternative wins a comparison. As a result, it will be quicker to reach a
  // choice, and more alternatives are more likely to win with more
  // alternatives.

  unsigned int n_alternatives = 2;
  unsigned int n_dimensions = 2;

  Eigen::VectorXd correct(n_alternatives);
  correct.setZero();

  Eigen::MatrixXd choice_set1;
  random_generate_choiceset(choice_set1, n_alternatives);

  Eigen::MatrixXd choice_set2(n_alternatives + 1, n_dimensions);
  // hardcoded for n_alternatives = 2 and n_dimensions = 2
  for (unsigned int j = 0; j < n_dimensions; j++) {
    choice_set2(0, j) = choice_set1(0, j);
    choice_set2(1, j) = std::numeric_limits<double>::quiet_NaN();
    choice_set2(2, j) = std::numeric_limits<double>::quiet_NaN();
  }
  choice_set2(1, 0) = choice_set1(1, 0);
  choice_set2(2, 1) = choice_set1(1, 1);

  std::vector<double> choice_probability1;
  std::vector<double> choice_probability2;

  model.predict(choice_probability1, choice_set1, parameter);
  model.predict(choice_probability2, choice_set2, parameter);

  correct(0) = choice_probability2.at(0);
  for (unsigned int i = 1; i < choice_probability2.size(); i++) {
    correct(1) += choice_probability2.at(i);
  }
  assess_result(correct, choice_probability1);
}

void test_predicting_effects(DbS::Model &model, DbS::Parameter &parameter) {
  std::cout << "Testing predictions: ";

  parameter.similarity_slope = 3.0;
  parameter.recognition_intercept = 0.1;
  parameter.recognition_slope = -50;
  parameter.threshold = 0.1;

  Eigen::MatrixXd choice_set;
  Eigen::VectorXd correct;

  unsigned int n_alternatives = 2;
  unsigned int n_dimensions = 2;

  choice_set.resize(n_alternatives, n_dimensions);
  correct.resize(n_alternatives);

  choice_set << -24000, 32, -16000, 24;
  correct << 0.5, 0.5;
  test_predicting_effect(model, parameter, choice_set, correct);

  n_alternatives = 3;
  choice_set.resize(n_alternatives, n_dimensions);
  correct.resize(n_alternatives);

  /* attraction effect */
  // with D
  choice_set << -24000, 32, -16000, 24, -27000, 29;
  correct << 0.468068, 0.270095, 0.261837;
  test_predicting_effect(model, parameter, choice_set, correct);

  // with F
  choice_set << -24000, 32, -16000, 24, -24000, 29;
  correct << 0.38137, 0.330316, 0.288314;
  test_predicting_effect(model, parameter, choice_set, correct);

  // with R
  choice_set << -24000, 32, -16000, 24, -27000, 32;
  correct << 0.42187, 0.284453, 0.293677;
  test_predicting_effect(model, parameter, choice_set, correct);

  // distant decoy
  choice_set << -24000, 32, -16000, 24, -28000, 28;
  correct << 0.516795, 0.245354, 0.237851;
  test_predicting_effect(model, parameter, choice_set, correct);

  // closer decoy
  choice_set << -24000, 32, -16000, 24, -26000, 30;
  correct << 0.377716, 0.318831, 0.303452;
  test_predicting_effect(model, parameter, choice_set, correct);

  /* compromise effect */
  choice_set << -24000, 32, -16000, 24, -32000, 40;
  correct << 0.402057, 0.294849, 0.303094;
  test_predicting_effect(model, parameter, choice_set, correct);

  /* similarity effect */
  choice_set << -24000, 32, -16000, 24, -17000, 25;
  correct << 0.375423, 0.315204, 0.309373;
  test_predicting_effect(model, parameter, choice_set, correct);

  n_alternatives = 5;
  choice_set.resize(n_alternatives, n_dimensions);
  correct.resize(n_alternatives);

  /* perceptual focus effect */
  choice_set << -24000, 32, -16000, 24, -28000, 24, -26000, 24, -18000, 24;
  correct << 0.40229333, 0.29908642, 0.00000000, 0.01900324, 0.27961701;
  test_predicting_effect(model, parameter, choice_set, correct);

  /* memory effect */
  std::vector<Eigen::VectorXd> memory_set;
  memory_set.resize(n_dimensions);

  n_alternatives = 4;
  choice_set.resize(n_alternatives, n_dimensions);
  correct.resize(n_alternatives);

  choice_set << -24000, 32, -16000, 24, -32000, 40, -8000, 16;
  memory_set.at(0).resize(n_alternatives);
  memory_set.at(0) << 32, 24, 40, 16;
  memory_set.at(1).resize(n_alternatives);
  memory_set.at(1) << -24000, -16000, -32000, -8000;
  correct << 0.297073, 0.280534, 0.220929, 0.201464;

  test_predicting_effect(model, parameter, choice_set, memory_set, correct);

  std::cout << " Pass" << std::endl;
}

void test_predicting_effect(DbS::Model &model, const DbS::Parameter &parameter,
                            const Eigen::MatrixXd &choice_set,
                            const Eigen::VectorXd &correct) {

  std::vector<double> choice_probability;
  model.predict(choice_probability, choice_set, parameter);

  assess_result(correct, choice_probability);
}

void test_predicting_effect(DbS::Model &model, const DbS::Parameter &parameter,
                            const Eigen::MatrixXd &choice_set,
                            const std::vector<Eigen::VectorXd> &memory_set,
                            const Eigen::VectorXd &correct) {

  std::vector<double> choice_probability;
  model.predict(choice_probability, choice_set, memory_set, parameter);

  assess_result(correct, choice_probability);
}

void assess_result(const Eigen::VectorXd &correct,
                   const std::vector<double> &computed,
                   const double tolerance) {

  assert(correct.size() == static_cast<int>(computed.size()));

  double err = 0;
  for (unsigned int i = 0; i < computed.size(); i++) {
    err += pow(correct(i) - computed.at(i), 2);
  }

  if (err > tolerance) {
    std::cout << " Failed" << std::endl;

    std::cout << "Correct: [";
    for (unsigned int i = 0; i < computed.size(); i++) {
      std::cout << correct(i) << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Computed: [";
    std::for_each(begin(computed), end(computed),
                  [](const double x) { std::cout << x << ", "; });
    std::cout << "]" << std::endl;

    throw std::logic_error("Failed");
  }

  std::cout << "." << std::flush;
}
