# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Fitting the Random Forest Regression the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X, y)

# prediting a new result
y_pred = regressor.predict(6.5)

# Visualising the Random Forest Regression results (for higher resolution and  smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='r')
plt.plot(X_grid, regressor.predict(X_grid), color='b')
plt.title('Truth or Bluff (Random Forest Regression )')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()