#%%

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from keras.datasets import mnist
import tensorflow.keras as kb
from tensorflow.keras import backend
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


from plotnine import *

from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score

from sklearn.linear_model import LinearRegression # Linear Regression Model
from sklearn.preprocessing import StandardScaler #Z-score variables

from sklearn.model_selection import train_test_split # simple TT split cv
from sklearn.model_selection import KFold # k-fold cv

from tensorflow.keras import regularizers
import os

#import categorical_embedder as ce







#%%

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')

print(df.isnull().sum().to_string())

print(df.head().to_string())

print(df.shape)

print(df.info())

#%%
df['medv'].describe()

import matplotlib.pyplot as plt

plt.matshow(df.corr())
plt.show()

import seaborn as sns
 
# checking correlation using heatmap
#Loading dataset
 
#plotting the heatmap for correlation
ax = sns.heatmap(df.corr(), annot=True)

#%%
feats = ["crim", 'nox', 'rm', 'rad', 'zn', 'indus', 'chas', 'dis', 'age', 'ptratio']
predict = "medv"

X = df[feats]
y = df[predict]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

z = StandardScaler()
X_train[feats] = z.fit_transform(X_train[feats])
X_test[feats] = z.transform(X_test[feats])


#%%

""" model = kb.Sequential([
    kb.layers.Dense(64, input_shape = [11]),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(128),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(128),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(256),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(128),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(128),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(64),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(1)
    
]) """

""" #structure of the model
model = kb.Sequential([
    kb.layers.Dense(64, input_shape = [11]),
    kb.layers.Dropout(0.5),
    kb.layers.Dense(128),
    kb.layers.Dropout(0.4),
    kb.layers.Dense(128),
    kb.layers.Dropout(0.4),
    kb.layers.Dense(256),
    kb.layers.Dropout(0.3),
    kb.layers.Dense(128),
    kb.layers.Dropout(0.3),
    kb.layers.Dense(128),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(64),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(1)
    
]) """


#%%


#%%


""" model = kb.Sequential([
    kb.layers.Dense(64, input_shape = [11]),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(128, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dropout(0.5),
    kb.layers.Dense(128, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dropout(0.5),
    kb.layers.Dense(128, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(64, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(64, activation='relu'),
    kb.layers.Dense(1)
    
]) """


#%%

model = kb.Sequential([
    kb.layers.Dense(64, input_shape = [11]),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(128, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(128, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(128, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(64, activation='relu'),
    kb.layers.BatchNormalization(),
    kb.layers.Dense(64, activation='relu'),
    kb.layers.Dense(1)
    
])

callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)]

#how to train the model
model.compile(loss = 'mean_squared_error',
            optimizer = 'adam',
            metrics = ['mean_absolute_percentage_error', 'mean_absolute_error'])

#fit the model (same as SKlearn)
model.fit(X_train, y_train, epochs = 200, validation_data = (X_test, y_test), callbacks = callbacks)