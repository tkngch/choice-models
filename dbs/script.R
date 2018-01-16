#!/usr/bin/env Rscript

library(Rcpp)
dyn.load("./libdbs_r.so")

parameter <- c(3, 0.1, -50, 1)
# parameter <- c(1.94, 0.40, -53.73, 0.52)

attraction_set <- list(c(-24000, 32), c(-16000, 24), c(-27000, 29))
attraction_effect <- .Call("evaluate_number_of_comparisons", attraction_set, parameter)
print(attraction_effect)

compromise_set <- list(c(-24000, 32), c(-16000, 24), c(-32000, 40))
compromise_effect <- .Call("evaluate_number_of_comparisons", compromise_set, parameter)
print(compromise_effect)

similarity_set <- list(c(-24000, 32), c(-16000, 24), c(-17000, 25))
similarity_effect <- .Call("evaluate_number_of_comparisons", similarity_set, parameter)
print(similarity_effect)
