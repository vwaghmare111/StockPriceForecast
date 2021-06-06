# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:49:06 2021

@author: vijay
"""
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, render_template,send_file,render_template_string
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import pandas as pd
import io
import base64
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error

######Code begins#####
app = Flask(__name__)

#Max period, i.e. all financial dates from inception date of a stock on secondary market
perd = 'max'

#Function to calculate significant lags from the result of Granger Causality Test
def max_lag(data):
    f_test = []
    chi_2 = []
    lr_test = []
    par_f = []
    key = []
    for i in range(1,len(data.keys())):
        keys = np.repeat(i,4)
        ftest = data[i][0]['ssr_ftest'][1]
        chi2 = data[i][0]['ssr_chi2test'][1]
        lrt = data[i][0]['lrtest'][1]#likelihood test ratio
        par_ftest = data[i][0]['params_ftest'][1]
    
        f_test.append(ftest)
        chi_2.append(chi2)
        lr_test.append(lrt)
        par_f.append(par_ftest)
        key.append(keys)
    
    keys_array = np.array(key).flatten()
    pvalues = np.column_stack([f_test,chi_2,lr_test,par_f]).flatten()
    df = pd.DataFrame(keys_array,columns=['keys'])
    df['pvalues'] = pvalues
    lag = df['keys'][df['pvalues']==df['pvalues'].min()].values[0]
    
    return lag

#Function to calculate business days given start date and number of days to be added
def add_business_days(from_date, ndays):
    business_days_to_add = abs(ndays)
    current_date = from_date
    sign = ndays/abs(ndays)
    while business_days_to_add > 0:
        current_date += timedelta(sign * 1)
        weekday = current_date.weekday()
        if weekday >= 5: # sunday = 6
            continue
        business_days_to_add -= 1
    return current_date

#Function dynamically calculate forecast dates given original datas last lag
def add_trading_days(data,lag):
    dates_idx = data[-lag:].index
    future_dates = []
    for i in range(len(dates_idx)):
        next_date = add_business_days(dates_idx[i],lag).strftime('%Y-%m-%d')
        future_dates.append(next_date)
    return future_dates

#Declaring variables to be used as global variables in a function
stockName = " "
pred_train = pd.DataFrame()
pred_test = pd.DataFrame()
forecast = pd.DataFrame()
X_train = pd.DataFrame()
X_test = pd.DataFrame()
Y_train = pd.DataFrame()
Y_test = pd.DataFrame()
prediction_df_ts = pd.DataFrame()
train_mae = 0
test_mae = 0

@app.route('/', methods=['GET', 'POST'])
def home():
    global stockName,pred_train,pred_test,forecast,X_train,X_test,Y_train,Y_test,train_mae,test_mae,prediction_df_ts
    if request.method == 'POST':        
        stockName = request.form.get('stock')
        daily_data = yf.download(tickers=stockName,period=perd,interval='1d')
        
        #Replacing inf with nan
        daily_data_new = daily_data.replace(-np.inf, np.nan)
        
        #Replacing nan with previous value
        daily_data_new.fillna(method='bfill', inplace=True)
        cl_op = grangercausalitytests(daily_data_new[['Close','Open']].diff(1).fillna(0),5,verbose=False)
        cl_hi = grangercausalitytests(daily_data_new[['Close','High']].diff(1).fillna(0),5,verbose=False)
        cl_lw = grangercausalitytests(daily_data_new[['Close','Low']].diff(1).fillna(0),5,verbose=False)
        
        #Granger casuality test max lag
        cl_op_lag = max_lag(cl_op)
        cl_hi_lag = max_lag(cl_hi)
        cl_lw_lag = max_lag(cl_lw)
        
        #Ideal lags
        lags = int(np.round(np.mean([cl_op_lag,cl_hi_lag,cl_lw_lag])))
        
        #Pre-processing data as per lags
        X = daily_data_new.drop(['Adj Close','Volume'],axis=1)
        Y = pd.DataFrame(daily_data_new['Close'])
        X_new = X[:-lags]
        Y_new = Y[lags:]
        
        # split the data into train and test
        X_train,X_test = train_test_split(X_new, train_size=0.80, shuffle=False)
        
        #Creating X and Y
        X_train = X_train.rename(columns={"Close":'Close_lagged'})
        X_test = X_test.rename(columns={"Close":'Close_lagged'})
        Y_train = Y_new[:int(X_train.shape[0])]
        Y_test = Y_new[int(X_train.shape[0]):]
        
        #Initializing and fitting Linear Regression model
        lr_model = linear_model.LinearRegression()
        lr_model.fit(X_train,Y_train)
        
        #Predicting train and test dataset
        pred_train = pd.DataFrame(lr_model.predict(X_train.values),index=Y_train.index,columns=['Predicted'])
        pred_test = pd.DataFrame(lr_model.predict(X_test.values),index=Y_test.index,columns=['Predicted'])
        
        #Actual vs predicted values for unseen (Test) data
        prediction_df_ts = pd.DataFrame(Y_test.values,columns = (['Actual']),index=Y_test.index)
        prediction_df_ts['Predicted'] = pred_test.values.copy()
        
        #Residual of Test prediction
        residual = prediction_df_ts['Actual']-prediction_df_ts['Predicted']        
        
        #Calculating MAE for train and test data
        train_mae = np.round(mean_absolute_error(Y_train,pred_train),3)
        test_mae = np.round(mean_absolute_error(Y_test,pred_test),3)

        #Calculating forecast days
        forecast_days = add_trading_days(X,lags)
        
        #Last lag data as X variables to forecast future trading days based on number of lags
        input_days = X[-lags:]        
        
        #Forecasting
        forecast = pd.DataFrame(lr_model.predict(input_days),index=forecast_days,columns=['Forecast'])
        forecast['Lower'] = forecast['Forecast'].values-residual.std()
        forecast['Upper'] = forecast['Forecast'].values+residual.std()
        forecast = forecast.round(3)
        #print(forecast.to_html())          
    return render_template("index.html", tables=[forecast.to_html(classes='table')], titles=forecast.columns.values, train_mae=train_mae,test_mae=test_mae,stockName=stockName)

    
#Function to plot prediction and forecast
@app.route('/plots')
def plots():
    fig, ax= plt.subplots(2, 1)
    prediction_df_ts.plot(ax=ax[0],title="Test Data Prediction",figsize=(10,5)).legend(loc='best')
    forecast.plot(ax=ax[1],title="Forecast",figsize=(10,5)).legend(loc='best')
    canvas = FigureCanvas(fig)
    img = io.BytesIO()
    plt.tight_layout()    
    plt.savefig(img,bbox_inches='tight',pad_inches=0,dpi=800)    
    img.seek(0)
    return send_file(img,mimetype='img/png')

#Function to disable caching
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
    

