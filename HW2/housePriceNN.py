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
cwd = os.getcwd()


df = pd.read_csv(cwd + '/data/train.csv')

print(df.isnull().sum().to_string())

print(df.head().to_string())

print(df.shape)


#%%

#Makes new variable that will make more sense for NN to understand
df['yearRemod'] = df['YearBuilt'] - df['YearRemodAdd']

df['LotFrontage'] = df['LotFrontage'].fillna(0)

df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

#%%


contin = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'MasVnrArea', 
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
        'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 
        'Fireplaces', 'GarageCars', 'GarageArea', 'PoolArea', 'WoodDeckSF', 'OpenPorchSF',
        'yearRemod']


ordinal = ['OverallQual', 'OverallCond']

predict = "SalePrice"

X = df[contin]
y = df[predict]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

""" embedding_info = ce.get_embedding_info(X)
X_encoded,encoders = ce.get_label_encoded_data(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded,y)
embeddings = ce.get_embeddings(X_train, y_train, categorical_embedding_info=embedding_info, 
                            is_classification=True, epochs=100,batch_size=256) """

z = StandardScaler()
X_train[contin] = z.fit_transform(X_train[contin])
X_test[contin] = z.transform(X_test[contin])



#%%

#%%


feats = ['LotFrontage', 'LotArea', 'PoolArea']

predict = "SalePrice"

X = df[feats]
y = df[predict]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


z = StandardScaler()
X_train[feats] = z.fit_transform(X_train[feats])
X_test[feats] = z.transform(X_test[feats])


#%%
# build structure of the model
model = kb.Sequential([
    kb.layers.Dense(64, input_shape =[4]), #input
    kb.layers.Dropout(0.5),
    kb.layers.Dense(64),
    kb.layers.Dense(64),
    kb.layers.Dropout(0.3),
    kb.layers.Dense(64),
    kb.layers.Dense(64),
    kb.layers.Dropout(0.2),
    kb.layers.Dense(64),
    kb.layers.Dense(64),
    kb.layers.Dense(1) #output
])

# compile model
model.compile(loss="mean_squared_error", optimizer= 'adam',
	metrics=["accuracy"])

#fit the model (same as SKlearn)
model.fit(X_train,y_train, epochs = 10, validation_data=(X_test, y_test))



#%%
model = kb.Sequential([
    kb.layers.Dense(64, input_shape = [3]),
    kb.layers.Dense(64),
    kb.layers.Dense(64),
    kb.layers.Dense(1)
    
])

# compile model
model.compile(loss="mean_squared_error", optimizer='rmsprop',
	metrics=["accuracy"])

#fit the model (same as SKlearn)
model.fit(X_train,y_train, epochs = 10, validation_data=(X_test, y_test))




#%%
df = pd.read_csv("https://raw.githubusercontent.com/cmparlettpelleriti/CPSC392ParlettPelleriti/master/Data/Music_data.csv")
feats = ["danceability", "energy", "loudness"]
predict = "valence"

X = df[feats]
y = df[predict]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

z = StandardScaler()
X_train[feats] = z.fit_transform(X_train[feats])
X_test[feats] = z.transform(X_test[feats])

# Regression

#%%
#structure of the model
model = kb.Sequential([
    kb.layers.Dense(1, input_shape = [2]),
    kb.layers.Dense(1),
    
])

#how to train the model
model.compile(loss = 'mean_squared_error',
              optimizer = kb.optimizers.SGD())

#fit the model (same as SKlearn)
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))

#%%

feats = ['LotArea', 'PoolArea']
predict = "SalePrice"

X = df[feats]
y = df[predict]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

z = StandardScaler()
X_train[feats] = z.fit_transform(X_train[feats])
X_test[feats] = z.transform(X_test[feats])


#%%

bostondf = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv')


#%%
feats = ["crim", 'nox', 'rm', 'rad']
predict = "medv"

X = bostondf[feats]
y = bostondf[predict]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

z = StandardScaler()
X_train[feats] = z.fit_transform(X_train[feats])
X_test[feats] = z.transform(X_test[feats])

# Regression


#%%
#structure of the model
model = kb.Sequential([
    kb.layers.Dense(64, input_shape = [4]),
    kb.layers.Dense(64),
    kb.layers.Dense(64),
    kb.layers.Dense(64),
    kb.layers.Dense(1)
    
])

#how to train the model
model.compile(loss = 'mean_squared_error',
              optimizer = 'adam')

#fit the model (same as SKlearn)
model.fit(X_train, y_train, epochs = 10, validation_data = (X_test, y_test))