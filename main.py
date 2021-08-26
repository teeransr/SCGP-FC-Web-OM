import numpy as np
import pandas as pd
import datetime as dt
from azure.storage.blob import BlobServiceClient
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA as ar
import threading as th
import warnings
import time

class Projection_Om:
    # ======================= Function =======================
    def __init__(self):
        self.STORAGEACCOUNTURL = "https://tpcomctrlmstoragedev.blob.core.windows.net"
        self.STORAGEACCOUNTKEY = "sC3rTzpCCefCKa80UY2+U8F76O8esw92NrQMa5TyOSeXb81Y3OKvp6MdCEY2hCKB32ip1mYgAhvC+T9SDFZ+dQ=="
        self.CONTAINERNAME = "scgp-fc-web-python"
        self.blob_service_client_instance = BlobServiceClient(account_url=self.STORAGEACCOUNTURL, credential=self.STORAGEACCOUNTKEY)

        self.VALID_COLUMNS = ['Year', 'Month', 'Element', 'Volume']
        self.MONTHS = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}


        self.MODELS = {
            0:self.LinearReg, 
            1:self.CurveFitting, 
            2:self.SMA,
            3:self.EMA, 
            4:self.ExpHW1, 
            5:self.ARIMA}

        self.ModelNames = {
            0:'Linear Regression', 
            1:'Curve Fitting', 
            2:'Simple Moving Average', 
            3:'Exponential Smoothing',
            4:'Holt-Winters Method', 
            5:'ARIMA'
            }

        # constants
        self.THRESHOLD = 6
        self.INACTIVE_MONTH=6
        self.FORECAST_PERIODS=2
        self.MA_PERIODS = 3
        self.MA_SELECT = False
        self.EMA_ALPHA = 0.9
        self.EMA_BETA = 0.9
        self.SEASONAL = 12
        self.HW_ALPHA = 0.9
        self.HW_BETA = 0.9
        self.HW_GAMMA = 0.9
        self.CF_DEGREE = 3
        self.ARIMA_P = 1
        self.ARIMA_D = 1
        self.ARIMA_Q = 1

    # ========================================================

    def input_data(self,blob_name):

        LOCALFILENAME = f"./inbound/{blob_name}"

        blob_client_instance = self.blob_service_client_instance.get_blob_client(
            self.CONTAINERNAME, blob_name, snapshot=None)

        with open(LOCALFILENAME, "wb") as my_blob:
            blob_data = blob_client_instance.download_blob()
            blob_data.readinto(my_blob)
        dataframe_blobdata = pd.read_csv(LOCALFILENAME,sep='|')
        return dataframe_blobdata

    # ========================================================
    def LinearReg(self):
        model = LinearRegression(fit_intercept=True, normalize=True,copy_X=True, n_jobs=1)
        fitted = model.fit(X=trainx, y=trainy)
        pred = fitted.predict(pd.DataFrame(dfIndices[i][1]))
    
    def CurveFitting(self):
        print('CurveFitting')

    def SMA(self, dfData, dfIndices, isPredict):
        #choose whether it uses the selected period or not
        if self.MA_SELECT:
            pr = range(MA_PERIODS,MA_PERIODS+1)
            return self.SimpleMovingAvg(dfData, dfIndices, isPredict, pr)
        else:
            pr = range(3,6)
            return self.SimpleMovingAvg(dfData, dfIndices, isPredict, pr)
    
    def ExpHW1(self):
        print('ExpHW1')

    # ========================================================

    def EMA(self, dfData, dfIndices, isPredict):
        
        if not isPredict:
        #train and test for the optimal alpha & beta
            acc = float('inf')
            for a in range(1,10,2):
                alpha = a/10
                for b in range(1,10,2):
                    beta = b/10
                    pacc = self.ExpoSmth(dfData, dfIndices, False, alpha, beta)
                    if pacc < acc:
                        acc = pacc
                        self.EMA_ALPHA = alpha
                        self.EMA_BETA = beta
            return acc
        else:
            # return prediction
            return self.ExpoSmth(dfData, dfIndices, True,self.EMA_ALPHA, self.EMA_BETA)
    
    # ========================================================
    
    def ARIMA(self):
        print('ARIMA')

    

    # ========================================================

    def output_data(self,blob_output_name):

        blob_client = self.blob_service_client_instance.get_blob_client(
            container=self.CONTAINERNAME, blob=blob_output_name)

        LOCALFILENAME = f"./outbound/{blob_output_name}"

        with open(LOCALFILENAME, "rb") as data:
            blob_client.upload_blob(data)

# ======================= Process =======================
if __name__=='__main__':

    obj_projection = Projection_Om()
    df_raw = obj_projection.input_data('historical_sales_paper.txt')
    