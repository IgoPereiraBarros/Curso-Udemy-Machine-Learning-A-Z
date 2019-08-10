# -*- coding: utf-8 -*-

# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
df = pd.read_csv('Ads_CTR_Optimisation.csv')

# Importing random selection
import random
N = 10000
d = 10
ads_selected = []
total_selected = 0

for n in range(N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = df.values[n, ad]
    total_selected = total_selected + reward
    
# Visualizing the results
plt.hist(ads_selected, bins=14, alpha=0.4, color='g')
plt.title('Histogram of Random Selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()