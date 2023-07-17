# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 20:29:28 2023

@author: alima
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, levene, ttest_ind, f_oneway, chi2_contingency, pearsonr
from math import sqrt
import seaborn as sns

df = pd.read_csv('C:/Career/Learning/IBM Statistics/boston_housing.csv')


##############
### TASK 4 ###
##############

### Part 1
ax = sns.boxplot(y = "MEDV", data = df)
ax.set_ylabel("Median value of owner-occupied \nhomes in $1000's")
plt.show()


### Part 2
ax = sns.countplot(x = 'CHAS', data = df)
ax.set_xlabel("Bounding Charles River?")
ax.set_ylabel("No. of Homes")
ax.set_xticklabels(["No", "Yes"])
plt.show()


### Part 3
df.loc[(df['AGE'] <= 35), 'age_group'] = '35 years and younger'
df.loc[(df['AGE'] > 35) & (df['AGE'] < 70), 'age_group'] = 'between 35 and 70'
df.loc[(df['AGE'] >= 70), 'age_group'] = '70 years and older'

box_order= ['35 years and younger', 'between 35 and 70', '70 years and older']

ax = sns.boxplot(y = "MEDV", x = 'age_group', data = df, order = box_order)
ax.set_ylabel("Median value of owner-occupied \nhomes in $1000's")
ax.set_xlabel("Age Groups", size = 15)
plt.show()

### Part 4

ax = sns.scatterplot(x = "INDUS", y = "NOX", data = df)
ax.set_ylabel("NOx Concentrations \n(in parts per 10 million)")
ax.set_xlabel("Proportion of Non-retail Business Acres per Town")
ax.set_yticks(np.arange(0.2, 1.2, 0.2))
plt.show()

### Part 5
ax = sns.histplot(df['PTRATIO'],
                  bins=20,
                  kde=True,
                  # hist_kws={"color": "green", "linewidth": 15,'alpha':1},
                  color='red'
                  )
ax.set(xlabel='Distribution of Pupil-Teacher Ratio', ylabel='Frequency')


##############
### TASK 5 ###
##############


