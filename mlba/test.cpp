// Copyright (C) 2015 Takao Noguchi (tkngch@runbox.com)

#include "test.hpp"

int main(void) {

  MLBA::Model model;
  MLBA::Parameter parameter;

  // test_effects(model, parameter);
  test_predicting_effects(model, parameter);
  test_predicting_effects2(model, parameter);

  return 0;
}

void test_predicting_effects(MLBA::Model &model, MLBA::Parameter &parameter) {

  parameter.m = 22.30;
  parameter.lambda1 = 0.193;
  parameter.lambda2 = 0.278;
  parameter.i0 = 13.39;

  Eigen::MatrixXd M(3, 2);
  Eigen::VectorXd correct(3);

  int res = 0;

  M << 23.8000, 45.9500, 45.7500, 24.0000, 17.0000, 45.9500;
  correct << 0.50553, 0.35964, 0.13483;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 23.8000, 45.9500, 45.7500, 24.0000, 45.7500, 17.2000;
  correct << 0.35706, 0.50774, 0.13520;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 25.5000, 45.2000, 35.2000, 35.5000, 24.0000, 46.7000;
  correct << 0.17152, 0.70840, 0.12008;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 25.5000, 45.2000, 35.2000, 35.5000, 37.1000, 33.6000;
  correct << 0.10961, 0.51157, 0.37882;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 44.7000, 26.0000, 35.2000, 35.5000, 46.0000, 24.7000;
  correct << 0.17794, 0.69235, 0.12971;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 44.7000, 26.0000, 35.2000, 35.5000, 34.0000, 36.7000;
  correct << 0.10923, 0.48951, 0.40126;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 73.3500, 36.6500, 53.0500, 56.9500, 43.0500, 66.9500;
  correct << 0.23027, 0.61005, 0.15968;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 53.0500, 56.9500, 43.0500, 66.9500, 33.1000, 76.9000;
  correct << 0.707992, 0.235281, 0.056727;
  res += test_predicting_effect(model, parameter, M, correct);

  if (res == 0) {
    std::cout << " Pass." << std::endl;
  }
}

void test_predicting_effects2(MLBA::Model &model, MLBA::Parameter &parameter) {

  parameter.m = 12.30;
  parameter.lambda1 = 0.133;
  parameter.lambda2 = 0.578;
  parameter.i0 = 18.39;

  Eigen::MatrixXd M(3, 2);
  Eigen::VectorXd correct(3);

  int res = 0;

  M << 23.8000, 45.9500, 45.7500, 24.0000, 17.0000, 45.9500;
  correct << 0.40085, 0.42540, 0.17375;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 23.8000, 45.9500, 45.7500, 24.0000, 45.7500, 17.2000;
  correct << 0.42356, 0.40202, 0.17442;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 25.5000, 45.2000, 35.2000, 35.5000, 24.0000, 46.7000;
  correct << 0.23633, 0.56914, 0.19454;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 25.5000, 45.2000, 35.2000, 35.5000, 37.1000, 33.6000;
  correct << 0.23440, 0.40202, 0.36358;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 44.7000, 26.0000, 35.2000, 35.5000, 46.0000, 24.7000;
  correct << 0.24048, 0.55756, 0.20196;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 44.7000, 26.0000, 35.2000, 35.5000, 34.0000, 36.7000;
  correct << 0.23489, 0.39519, 0.36992;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 73.3500, 36.6500, 53.0500, 56.9500, 43.0500, 66.9500;
  correct << 0.32229, 0.41724, 0.26047;
  res += test_predicting_effect(model, parameter, M, correct);

  M << 53.0500, 56.9500, 43.0500, 66.9500, 33.1000, 76.9000;
  correct << 0.56310, 0.28756, 0.14934;
  res += test_predicting_effect(model, parameter, M, correct);

  if (res == 0) {
    std::cout << " Pass." << std::endl;
  }
}

int test_predicting_effect(MLBA::Model &model, const MLBA::Parameter &parameter,
                           Eigen::MatrixXd &M, const Eigen::VectorXd &correct) {

  std::cout << "." << std::flush;
  std::vector<double> choice_probability;
  M /= 10.0;

  model.predict(choice_probability, M, parameter);
  int res = assess_results(correct, choice_probability);

  return res;
}

void test_effects(MLBA::Model &model, const MLBA::Parameter &parameter) {

  Eigen::VectorXd correct_rate(3);
  Eigen::MatrixXd M(3, 2);

  int res = 0;

  M << 23.8000, 45.9500, 45.7500, 24.0000, 17.0000, 45.9500;
  correct_rate << 15.417, 14.457, 12.237;
  res += test_effect(model, parameter, M, correct_rate);

  M << 23.8000, 45.9500, 45.7500, 24.0000, 45.7500, 17.2000;
  correct_rate << 14.439, 15.432, 12.243;
  res += test_effect(model, parameter, M, correct_rate);

  M << 25.5000, 45.2000, 35.2000, 35.5000, 24.0000, 46.7000;
  correct_rate << 12.615, 16.396, 11.936;
  res += test_effect(model, parameter, M, correct_rate);

  M << 25.5000, 45.2000, 35.2000, 35.5000, 37.1000, 33.6000;
  correct_rate << 11.586, 15.110, 14.272;
  res += test_effect(model, parameter, M, correct_rate);

  M << 44.7000, 26.0000, 35.2000, 35.5000, 46.0000, 24.7000;
  correct_rate << 12.640, 16.253, 12.027;
  res += test_effect(model, parameter, M, correct_rate);

  M << 44.7000, 26.0000, 35.2000, 35.5000, 34.0000, 36.7000;
  correct_rate << 11.557, 14.954, 14.400;
  res += test_effect(model, parameter, M, correct_rate);

  M << 73.3500, 36.6500, 53.0500, 56.9500, 43.0500, 66.9500;
  correct_rate << 13.550, 16.242, 12.756;
  res += test_effect(model, parameter, M, correct_rate);

  M << 53.0500, 56.9500, 43.0500, 66.9500, 33.1000, 76.9000;
  correct_rate << 16.817, 13.661, 11.059;
  res += test_effect(model, parameter, M, correct_rate);

  if (res == 0) {
    std::cout << " Pass." << std::endl;
  }
}

int test_effect(MLBA::Model &model, const MLBA::Parameter &parameter,
                Eigen::MatrixXd &M, const Eigen::VectorXd &correct_rate) {

  std::cout << "." << std::flush;
  std::vector<double> drift_rate;
  M /= 10.0;

  model.compute(drift_rate, M, parameter);
  int res = assess_results(correct_rate, drift_rate);

  return res;
}

int assess_results(const Eigen::VectorXd &correct_rate,
                   const std::vector<double> &drift_rate) {

  assert(correct_rate.size() == static_cast<int>(drift_rate.size()));

  auto print_vector = [](const std::vector<double> &vector) {
    std::for_each(begin(vector), end(vector),
                  [](const double &x) { std::cout << x << ", "; });
  };

  double err = 0;
  for (unsigned int i = 0; i < correct_rate.size(); i++) {
    err += pow(correct_rate(i) - drift_rate.at(i), 2);
  }

  int out = 0;
  double threshold = 1e-10;
  if (err > threshold) {

    std::cout << " Failed!" << std::endl;

    std::cout << "Correct: [" << correct_rate << "]" << std::endl;

    std::cout << "Computed: [";
    print_vector(drift_rate);
    std::cout << "]" << std::endl;

    out = 1;
  }

  return out;
}
