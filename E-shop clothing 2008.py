# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:25:04 2021

@author: HP630
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 17:25:04 2021

@author: HP630
"""
###########################################################################
###########################################################################
###                                                                     ###
###                     ESHOPS CCLOTHINGS   PROJECT                     ###
###                          LINEAR REGRESSION                          ###
###                                                                     ###
###########################################################################
###########################################################################
###*                        Loading Packages
###*                        ----------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
##===============================================================
##                     Reading in the data                     ==
##===============================================================

pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)
#chemin ='D:\Users\HP630\Downloads\Compressed\E-shoclothing2008.csv'

data = pd.read_csv('2E-shoclothing2008.csv', index_col=0, encoding = "ISO-8859-1")

#url = 'https://github.com/aissakadda/RTest-Project1/blob/main/2E-shoclothing2008.csv'
#data = pd.read_csv(url, index_col=0, encoding = "ISO-8859-1")
##================================================================
##                EDA: Exploratory Data Analysis                ==
##================================================================

print("Shape Data orgine",data.shape)
df = data.copy()
print("Data Copycolumns",df.columns)
print("Shape Data Copy",df.shape)
print("Data describe",df.describe())
print("Pourcentage des response Oui/Non pour le prix est superieur a la moyenne")
df['price2'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage COUNTRY")
df['country'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage des Mois")
df['month'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage des page1")
df['page1'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage des colour")
df['colour'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage des mph")
df['mph'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage des price")
df['price'].value_counts(normalize=True).plot.pie()
plt.show()

print("Pourcentage des page")
df['page'].value_counts(normalize=True).plot.pie()
plt.show()

##***************************************************************
##                    Study the correlation  var Price         **
##***************************************************************

print(df.corr()['price'].sort_values())
plt.figure(figsize=(10,10))
sns.heatmap(df.corr())
plt.show()


################################################################
##                        Joint Graphs                        ##
################################################################


# PRICE VS page1
sns.lmplot(x="page1", y="price", col="price2", hue="price2", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})

# PRICE VS mph
sns.lmplot(x="mph", y="price", col="price2", hue="price2", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})

# PRICE VS month
sns.lmplot(x="mph", y="price", col="price2", hue="price2", data=df,
           col_wrap=2, ci=None, palette="muted", height=4,
           scatter_kws={"s": 50, "alpha": 1})
plt.figure(figsize=(10,10))
sns.heatmap(pd.crosstab(df['price2'], df['page1']), annot=True, fmt='d')
plt.show()

###*************************************************************************
###*************************************************************************
###                                                                      ***
###                          SPLITTING THE DATA                          ***
###                        TRAINING AND TEST SETS                        ***
###                                                                      ***
###*************************************************************************
###*************************************************************************
###*

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

y = df['price']
X = df.drop(['page2','price'], axis=1)

#  testset =0,2 trainset=0,8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print('Train set X:', X_train.shape)
print('Train  set y:', y_train.shape)
print('Test  set X:', X_test.shape)
print('Test set y:', y_test.shape)


model = LinearRegression()
model.fit(X, y) # entrainement du modele

print('train score:', model.score(X_train, y_train))
print('test score:', model.score(X_test, y_test))

print('predict model',model.predict(X))


# parameters mean_squared_error  squaredbool default=True If True returns 
# MSE value  if False returns RMSE value uniform_average
# Errors of all outputs are averaged with uniform weight
mse = mean_squared_error(y_test, model.predict(X_test))
print("The mean squared error (MSE) on test set: ",mse)
rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
print("The R-mean squared error (RMSE) on test set: ",rmse)
mae = mean_absolute_error(y_test, model.predict(X_test), multioutput='raw_values')
print("The Mean Absolute_Error(MAE) on test set: ",mae)


