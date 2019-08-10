# -*- coding: utf-8 -*-

# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importing the dataset
df = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
for i in range(len(df)):
    transactions.append([str(df.values[i, j]) for j in range(len(df.columns))])

# Training Apriori on the dataset
from apyori import apriori
"""
regras:
min_support --> 1 produto por dia, sendo no máximo 3 dias * número total de 
                semanas / número total de transações
            --> 3 * 7 / 7500 = 0.0028

min_confidence --> 20 % de confiança

min_lift --> None

min_length --> determina que queremos no mímino dois produtos neste modelo

"""
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Visualization the results
results = list(rules)