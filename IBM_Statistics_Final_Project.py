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
import statsmodels.api as sm


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

### Part 1
# Hyothesis: There is no influece of bounding to the river on the asset values
p_q1 = ttest_ind(df[df['CHAS'] == 0]['MEDV'],
                 df[df['CHAS'] == 1]['MEDV'], equal_var = True)[1]

# Conclusion: As p < 0.05, we reject the null hyothesis, therefore, bounding to the river significantly influence the prices
   

### Part 2
# Hyothesis: There is no influece by the age range on the asset values
p_q2 = f_oneway(df[(df['AGE'] <= 35)]['MEDV'],
                df[(df['AGE'] > 35)& (df['AGE'] < 70)]['MEDV'],
                df[(df['AGE'] >= 70)]['MEDV'])[1]

# Conclusion: As p < 0.05, we reject the null hyothesis, therefore, age range of the asset significantly influences its prices

### Part 3
# Hyothesis: There is no correlation between the Nox concentration and the weighted distance to the Boston employment centres
p_q3 = pearsonr(df['INDUS'], df['NOX'])[1]
p_r = pearsonr(df['INDUS'], df['NOX'])[0]

# Conclusions: There is a statsitically significant corelation between the NOx concentration and weighted distance to employment
# centres (p<0.05). This correlation is relatively strong due to 0.76 as the correlation coeffieicnt. 


### Part 4
# Hyothesis: There is no correlation between the home values and the weighted distance to the Boston employment centres
X = df['DIS']
y = df['MEDV']

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)
model.summary()

pearsonr(df['DIS'], df['MEDV'])

# Conclusions: There is a statsitically significant corelation between the home values and weighted distance to employment
# centres (p<0.05). However, this correlation is relatively weak due to r2 of 0.062 (r of 0.25). 

