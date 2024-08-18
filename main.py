# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:57:08 2024

@author: lol10
"""

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
#import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

#df = pd.read_csv('casas.csv')
colunas= ['tamanho', 'ano', 'garagem']

#colunas= ['tamanho', 'preco']
#df = df[colunas]

#X = df.drop('preco', axis=1)
#y = df['preco']

#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#modelo = LinearRegression()
#modelo.fit(X_train, y_train)

# Carregando o modelo diretamente
modelo = pickle.load(open('modelo.sav','rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'user'
app.config['BASIC_AUTH_PASSWORD'] = 'password'

basic_auth = BasicAuth(app)

@app.route('/')

def home():
    return "My first API"


@app.route('/sentimento/<text>')
@basic_auth.required
def sentimento(text):
    tb = TextBlob(text)
    polarity =  tb.sentiment.polarity
    return "polarity: {}".format(polarity)


#@app.route('/lr/<int:tamanho>')
#def lr(tamanho):
#    preco = modelo.predict([[tamanho]])
#    return str(preco)

@app.route('/lr/', methods=['POST'])
@basic_auth.required
def lr():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])


app.run(debug=True)