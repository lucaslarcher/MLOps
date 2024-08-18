# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 18:18:50 2024

@author: lol10
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('casas.csv')
colunas= ['tamanho', 'ano', 'garagem']

#colunas= ['tamanho', 'preco']
#df = df[colunas]

X = df.drop('preco', axis=1)
y = df['preco']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

pickle.dump(modelo, open('modelo.sav','wb'))
