// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include "test.hpp"

int main(void) {

  AAM::Model model;
  AAM::Parameter parameter;

  test_effects(model, parameter);

  return 0;
}

void test_effects(AAM::Model &model, AAM::Parameter &parameter) {

  int res = 0;

  Eigen::VectorXd correct_rate(3);
  Eigen::MatrixXd M(3, 2);

  // 3 alternatives
  M << 3, 7, 7, 3, 2, 6;
  correct_rate << 0.59950, 0.40050, 0.00000;
  res += test_effect(model, parameter, M, correct_rate);

  M << 8, 2, 5, 5, 2, 8;
  correct_rate << 0.33450, 0.32550, 0.34000;
  res += test_effect(model, parameter, M, correct_rate);

  M << 3, 7, 7, 3, 3.1, 6.9;
  correct_rate << 0.30700, 0.39250, 0.30050;
  res += test_effect(model, parameter, M, correct_rate);

  // 2 alternatives
  M.resize(2, 2);
  correct_rate.resize(2);

  M << 5, 5, 0, 10;
  correct_rate << 0.63000, 0.37000;
  res += test_effect(model, parameter, M, correct_rate);

  M << 3, 7, 7, 3;
  correct_rate << 0.56000, 0.44000;
  res += test_effect(model, parameter, M, correct_rate);

  // different parameter values
  M.resize(3, 2);
  correct_rate.resize(3);

  // d
  parameter.d = 0.99;
  M << 3, 7, 7, 3, 2, 6;
  correct_rate << 0.83000, 0.17000, 0.00000;
  res += test_effect(model, parameter, M, correct_rate);

  M << 8, 2, 5, 5, 2, 8;
  correct_rate << 0.100000, 0.850000, 0.050000;
  res += test_effect(model, parameter, M, correct_rate);

  M << 3, 7, 7, 3, 3.1, 6.9;
  correct_rate << 0.32950, 0.21750, 0.45300;
  res += test_effect(model, parameter, M, correct_rate);

  // e
  parameter.e = 1;
  M << 3, 7, 7, 3, 2, 6;
  correct_rate << 0.670000, 0.300000, 0.030000;
  res += test_effect(model, parameter, M, correct_rate);

  M << 8, 2, 5, 5, 2, 8;
  correct_rate << 0.19000, 0.59000, 0.22000;
  res += test_effect(model, parameter, M, correct_rate);

  M << 3, 7, 7, 3, 3.1, 6.9;
  correct_rate << 0.38500, 0.21050, 0.40450;
  res += test_effect(model, parameter, M, correct_rate);

  if (res == 0) {
    std::cout << " Pass." << std::endl;
  }
}

int test_effect(AAM::Model &model, const AAM::Parameter &parameter,
                const Eigen::MatrixXd &M, const Eigen::VectorXd &correct_rate) {

  std::cout << "." << std::flush;
  std::vector<double> choice_probability;

  model.predict(choice_probability, M, parameter);
  int res = assess_result(correct_rate, choice_probability);

  return res;
}

int assess_result(const Eigen::VectorXd &correct_rate,
                  const std::vector<double> &choice_probability) {

  assert(correct_rate.size() == static_cast<int>(choice_probability.size()));

  auto print_vector = [](const std::vector<double> &vector) {
    std::for_each(begin(vector), end(vector),
                  [](const double &x) { std::cout << x << ", "; });
  };

  double err = 0;
  for (unsigned int i = 0; i < correct_rate.size(); i++) {
    err += pow(correct_rate(i) - choice_probability.at(i), 2);
  }

  int out = 0;
  double threshold = 0.01;
  // std::cout << err << std::endl;
  if (err > threshold) {

    std::cout << " Failed!" << std::endl;

    std::cout << "Correct: [" << correct_rate << "]" << std::endl;

    std::cout << "Computed: [";
    print_vector(choice_probability);
    std::cout << "]" << std::endl;

    out = 1;
  }

  return out;
}
