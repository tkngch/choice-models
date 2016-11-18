// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include "test_mdft.hpp"

int main(void) {
  MDFT::Model model;
  MDFT::Parameter parameter;

  test_predicting_effects(model, parameter);
  test_simulating_effects(model, parameter);

  return 0;
}

void test_predicting_effects(MDFT::Model &model,
                             const MDFT::Parameter &parameter) {

  int res = 0;

  unsigned int n_alternatives = 3;
  unsigned int n_dimensions = 2;
  Eigen::MatrixXd M(n_alternatives, n_dimensions);
  Eigen::VectorXd correct(n_alternatives);

  M << 1, 3, 3, 1, 0.9, 3.1;
  correct << 0.29597, 0.39414, 0.30990;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 1.1, 2.9;
  correct << 0.30836, 0.39306, 0.29857;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 1.75, 2.25;
  correct << 0.30904, 0.30144, 0.38953;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 2, 2;
  correct << 0.29727, 0.29727, 0.40546;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 2.25, 1.75;
  correct << 0.30144, 0.30904, 0.38953;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 0.5, 2.5;
  correct << 0.79551, 0.20449, 0.00000;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 1.1, 2.5;
  correct << 0.97580, 0.02420, 0.00000;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 0.5, 0.5, 0.7, 0.7, 2, 2;
  correct << 0.00000, 0.01319, 0.98681;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 0.75, 0.75, 1, 1, 1.25, 1.25;
  correct << 0.00000, 0.15940, 0.84060;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 0.5, 0.5, 1, 1, 2, 2;
  correct << 0.00000, 0.00000, 1.00000;
  res += test_predicting_effect(model, parameter, M, correct);

  if (res == 0) {
    std::cout << " Pass";
  }
  std::cout << std::endl;
}

int test_predicting_effect(MDFT::Model &model, const MDFT::Parameter &parameter,
                           const Eigen::MatrixXd &M,
                           const Eigen::VectorXd &correct) {

  std::vector<double> choice_probability;
  model.predict(choice_probability, M, parameter);

  int res = assess_result(correct, choice_probability);
  return res;
}

void test_simulating_effects(MDFT::Model &model, MDFT::Parameter &parameter) {

  parameter.sig2 = 1;

  int res = 0;

  unsigned int n_alternatives = 3;
  unsigned int n_dimensions = 2;
  Eigen::MatrixXd M(n_alternatives, n_dimensions);
  Eigen::VectorXd correct(n_alternatives);

  M << 1, 3, 3, 1, 0.9, 3.1;
  correct << 0.2984, 0.3953, 0.3063;
  res += test_simulating_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 1.1, 2.9;
  correct << 0.3059, 0.3800, 0.3141;
  res += test_simulating_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 1.75, 2.25;
  correct << 0.2811, 0.2983, 0.4207;
  res += test_simulating_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 2, 2;
  correct << 0.2848, 0.2809, 0.4342;
  res += test_simulating_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 2.25, 1.75;
  correct << 0.3004, 0.2773, 0.4223;
  res += test_simulating_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 0.5, 2.5;
  correct << 0.5545, 0.4455, 0.0000;
  res += test_simulating_effect(model, parameter, M, correct);

  M << 1, 3, 3, 1, 1.1, 2.5;
  correct << 0.6199, 0.3735, 0.0066;
  res += test_simulating_effect(model, parameter, M, correct);

  M << 0.5, 0.5, 0.7, 0.7, 2, 2;
  correct << 0.0000, 0.0005, 0.9995;
  res += test_simulating_effect(model, parameter, M, correct);

  M << 0.75, 0.75, 1, 1, 1.25, 1.25;
  correct << 0.0000, 0.3255, 0.6743;
  res += test_simulating_effect(model, parameter, M, correct);

  M << 0.5, 0.5, 1, 1, 2, 2;
  correct << 0.0000, 0.0003, 0.9998;
  res += test_simulating_effect(model, parameter, M, correct);

  if (res == 0) {
    std::cout << " Pass";
  }
  std::cout << std::endl;
}

int test_simulating_effect(MDFT::Model &model, const MDFT::Parameter &parameter,
                           const Eigen::MatrixXd &M,
                           const Eigen::VectorXd &correct) {

  const double tolerance = 0.005;

  std::vector<double> choice_probability;
  model.simulate(choice_probability, M, parameter);

  int res = assess_result(correct, choice_probability, tolerance);
  return res;
}

int assess_result(const Eigen::VectorXd &correct,
                  const std::vector<double> &computed, const double tolerance) {

  assert(correct.size() == static_cast<int>(computed.size()));

  double err = 0;
  for (unsigned int i = 0; i < computed.size(); i++) {
    err += pow(correct(i) - computed.at(i), 2);
  }

  int res = 0;
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

    res = 1;
  } else {
    std::cout << "." << std::flush;
  }

  return res;
}
