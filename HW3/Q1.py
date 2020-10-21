# CSCE 633 - Machine Learning
# HW3 - Question 1
import pdb
import numpy as np
import pandas as pd

data = pd.read_csv('Q1_Data.csv', header=None)
data = data.to_numpy()

# calculate mean and variance
mean = np.sum(data) / len(data)
var = np.sum(np.power(data - mean, 2)) / len(data)

print('Mean: %.5f, Variance: %.5f' % (mean, var))