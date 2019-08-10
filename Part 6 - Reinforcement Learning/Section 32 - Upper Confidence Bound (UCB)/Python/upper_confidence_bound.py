# -*- coding: utf-8 -*-

# Importing the libreries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implemeting UCB
from math import sqrt, log

N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0

for n in range(N):
    ad = 0
    max_upper_bound = 0
    for i in range(d):
        if numbers_of_selections[i] > 0:
            average_reward = sum_of_rewards[i] / numbers_of_selections[i]
            delta_i = sqrt(3/2 * log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = df.values[n, ad]
    sum_of_rewards[ad] += reward
    total_reward += reward
    
# Visualizing the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()