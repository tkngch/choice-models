#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


alpha = 3
beta0 = 0.1
beta1 = 50
theta = 0.1

choiceSet = numpy.array([[-1200, 720], [-1000, 640]])
memorySet = numpy.array([[-1350, 1200], [-1250, 740]])
nAlternatives = choiceSet.shape[0]
values = numpy.vstack([choiceSet, memorySet])

# difference in values
difference = (
    choiceSet.T.reshape((choiceSet.shape[1], choiceSet.shape[0], 1)) -
    values.T.reshape((values.shape[1], 1, values.shape[0]))
)

# distance as fractions
distance = numpy.abs(
    difference /
    values.T.reshape((values.shape[1], 1, values.shape[0]))
)
# we do not consider self-comparison, so don't compute self-distance
index = numpy.arange(choiceSet.shape[0])
distance[:, index, index] = numpy.nan
print("distance\n", numpy.round(distance, 2), "\n\n")

# similarity
similarity = numpy.exp(- alpha * distance)
print("similarity\n", numpy.round(similarity, 2), "\n\n")
print("similarity.sum\n", numpy.round(numpy.nansum(similarity), 2), "\n\n")

# probability of evaluation
pEvaluation = numpy.nansum(similarity, axis=2) / numpy.nansum(similarity)
print("pEvaluation\n", numpy.round(pEvaluation, 2), "\n\n")

# probability to recognize a difference
pRecognise = sigmoid(beta1 * (distance - beta0)) * (difference > 0)
print("pRecognise\n", numpy.round(pRecognise, 2), "\n\n")

# probability to win a comparison
pWin = numpy.nanmean(pRecognise, axis=2)
print("pWin\n", numpy.round(pWin, 2), "\n\n")

# accumulation rate
accumulationRate = (pEvaluation * pWin).sum(axis=0)
print("Accumulation Rate\n", numpy.round(accumulationRate, 2), "\n\n")

# choice probability
if theta < 1. / nAlternatives:
    pChoice = accumulationRate / accumulationRate.sum()
    print("pChoice\n", numpy.round(pChoice, 2), "\n\n")
