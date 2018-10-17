'''
Created on Oct 17, 2018

@author: Agha Rameez
'''
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from flask import Flask, render_template  # From module flask import class Flask
from cookielib import debug
app = Flask(__name__)    # Construct an instance of Flask class for our webapp

@app.route('/')   # URL '/' to be handled by main() route handler
def main():
    df = pd.read_csv('DataSet-nfl_elo1.csv')
    df = pd.DataFrame(df)
    x=df[['season','neutral','elo1_pre','elo2_pre','elo_prob1','elo_prob2','elo1_post','elo2_post','score1']]
    y=df['score2']
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    return render_template('home.html', X_train=X_train.shape ,y_train=y_train.shape,X_test=X_test.shape,y_test= y_test.shape,score=model.score(X_test,y_test),title='NFL DATA' )
@app.route('/aids')
def aids():
    df = pd.read_csv('aids.csv')
    df = pd.DataFrame(df)
    x=df[['quarter','delay','dud','time']]
    y=df['y']
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    return render_template('home.html', X_train=X_train.shape ,y_train=y_train.shape,X_test=X_test.shape,y_test= y_test.shape,score=model.score(X_test,y_test),title='AID DATA' )
@app.route('/carprices')
def car():
    df = pd.read_excel('carprices.xlsx',sheet_name=0)
    df = pd.DataFrame(df)
    x=df[['Mileage','Age(yrs)']]
    y=df['Sell Price($)']
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2)
    lm = linear_model.LinearRegression()
    model = lm.fit(X_train, y_train)
    predictions = lm.predict(X_test)
    return render_template('home.html', X_train=X_train.shape ,y_train=y_train.shape,X_test=X_test.shape,y_test= y_test.shape,score=model.score(X_test,y_test),title='Car PRICES' )
if __name__ == '__main__':  # Script executed directly?
    app.run(debug=True)  # Launch built-in web server and run this Flask webapp
