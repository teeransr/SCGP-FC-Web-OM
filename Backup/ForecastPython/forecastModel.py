# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 23:58:19 2017

@author: suradiss
"""

import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
#import scipy, scipy.stats, scipy.optimize as opt
from statsmodels.tsa.arima_model import ARIMA as ar
import threading as th
import warnings
import time

VALID_Columns = ['Year', 'Month', 'Element', 'Volume']

Months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
          'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}


# constants
THRESHOLD = 6
INACTIVE_MONTH=6
FORECAST_PERIODS=2
MA_PERIODS = 3
MA_SELECT = False
EMA_ALPHA = 0.9
EMA_BETA = 0.9
SEASONAL = 12
HW_ALPHA = 0.9
HW_BETA = 0.9
HW_GAMMA = 0.9
CF_DEGREE = 3
ARIMA_P = 1
ARIMA_D = 1
ARIMA_Q = 1

#time series class
class TSA(object):
    
    #data preprocessing
    def summarize_data(self):
        dicdf = dict()
        for index,row in self.df_raw.iterrows():
            ind = (row[2], row[0], Months[row[1]])
            dat = row[3]
            if ind in dicdf:
                dicdf[ind]+=dat
            else:
                dicdf[ind]=dat
        
        tsdic = dict()
        self.Max_Date = dt.date.min
        for k in dicdf.keys():
            mmyy = dt.date(k[1],k[2],1)
            if self.Max_Date < mmyy:
                self.Max_Date=mmyy
            if k[0] not in tsdic:
                tsdic[k[0]] = {mmyy: dicdf[k]}
            else:
                tsdic[k[0]][mmyy] = dicdf[k]
        
        
        #filter too short ts
        new_tsdic = dict()
        for i in tsdic.keys():
            if len(tsdic[i]) >= THRESHOLD:
                new_tsdic[i]=tsdic[i]
                
        tsdic = new_tsdic
        
        # re-arrange ts by date and filter inactive sku
        new_tsdic = dict()
        for j in tsdic.keys():
            temp = tsdic[j]
            tsdic[j] = pd.DataFrame.from_dict(temp, orient='index')
            tsdic[j].columns = ['Vol']
            tsdic[j].sort_index(inplace=True)
            tt = tsdic[j]
            
            lastM = tt.index.max()
            fence = self.Max_Date + relativedelta(months=-INACTIVE_MONTH)
            # filter out inactive sku
            if lastM >= fence:
                new_tsdic[j] = tt.reindex(pd.date_range(start=tt.index.min(),
                                                      end=self.Max_Date,
                                                      freq='MS'))
                new_tsdic[j].fillna(0, inplace=True)
            
        return new_tsdic
    
    
    def LinearReg(self, dfData, dfIndices, isPredict):
        
        #build a model
        model = LinearRegression(fit_intercept=True, normalize=True,
                                 copy_X=True, n_jobs=1)
        
        #accList stores accuracy for each instance
        accList = list()
        instances = len(dfData)
        for i in range(instances):
            
            trainx = pd.DataFrame(dfIndices[i][0], columns=['id'])
            trainy = dfData[i][0]
            trainy = trainy.values.reshape(-1,1)
            fitted = model.fit(X=trainx, y=trainy)
            pred = fitted.predict(pd.DataFrame(dfIndices[i][1]))
            print(fitted.coef_, fitted.intercept_)
            #if it's not cross-validation, return prediction
            if isPredict:
                return pred
            #evaluate instances using RMSE
            testy = dfData[i][1]
            acc = np.sqrt(mean_squared_error(testy,pred))
            accList.append(acc)
        
        weights = [i/(instances*(instances+1)/2) for i in range(1,instances+1)]
        wacc = np.average(accList, weights=weights)
        return wacc
    
    
    def CurveFitting(self, dfData, dfIndices, isPredict):   
        #Find y
        def polyValue(coeff, x):
            polysum = 0.0
            deg = len(coeff)
            X=1
            for i in range(deg-1, -1, -1):
                polysum += coeff[i]*X
                X*=x
            return float(polysum)
        
        #Turn off rank warning
        warnings.simplefilter('ignore', np.RankWarning)
        
        #instances = len(dfData)
        if not isPredict:
            #Fit degree 2 and 3
            instances = len(dfData)
            global CF_DEGREE
            acc = float('inf')
            for d in range(2,4):
                accList = list()
                for i in range(instances):
                    trainx = dfIndices[i][0]
                    trainy = dfData[i][0]
                    try:
                        cfFitted = np.polyfit(trainx, trainy, d)
                    except:
                        #if there exists an error, return inf
                        return float('inf')
                    pred = list()
                    testx = dfIndices[i][1]
                    testy = dfData[i][1]
                    for j in range(2):
                        pred.append(polyValue(cfFitted,testx[j]))
                        
                    pred = pd.DataFrame(pred, columns=['id'])
                    iacc = np.sqrt(mean_squared_error(testy, pred))
                    accList.append(iacc)
                weights = [i/(instances*(instances+1)/2) for i in 
                           range(1,instances+1)]
                wacc = np.average(accList, weights=weights)
                if wacc < acc:
                    acc = wacc
                    CF_DEGREE = d
            return acc
        else:
            trainx = dfIndices[0][0]
            trainy = dfData[0][0]
            deg = CF_DEGREE
            coef = np.polyfit(trainx, trainy, deg)
            testx = dfIndices[0][1]
            pred = list()
            for k in range(FORECAST_PERIODS):
                pred.append(polyValue(coef, testx[k]))
            return pred
    
    def SimpleMovingAvg(self, dfData, dfIndices, isPredict, periods):
        #inner method to calculate projection
        def rolling_avg(series, n, forecast_p):
            if n>len(series):
                return None
            pred = list()
            for k in range(forecast_p):
                ss = 0
                if k-n < 0:
                    ss = float(series[k-n:].sum())
                    if k > 0:
                        ss += sum(pred[0:k+1])
                else:
                    ss = sum(pred[k-n:k+1])
                pred.append(ss/n)
                
            return pred
        
        instances = len(dfData)
        global MA_PERIODS
        
        if not isPredict:
            acc = float('inf')
            best = periods[0]
            for j in periods:
                accList = list()
                usedN = 0
                for i in range(instances):
                    trainy = dfData[i][0]
                    trn = rolling_avg(trainy, j, 2)
                    if trn == None:
                        continue
                    # test and find accuracy
                    usedN += 1
                    trn = pd.DataFrame(trn, columns=['Vol'])
                    testy = dfData[i][1]
                    iacc = np.sqrt(mean_squared_error(testy,trn))
                    accList.append(iacc)
                if len(accList) == 0:
                    continue
                weights = [i/(usedN*(usedN+1)/2) for i in range(1,usedN+1)]
                wacc = np.average(accList, weights=weights)
                print('n=',j)
                print('acc=',wacc)
                if wacc < acc:
                    acc = wacc
                    best = j
            MA_PERIODS = best
            return acc
        else:
            trainy = dfData[0][0]
            pred = rolling_avg(trainy, MA_PERIODS, FORECAST_PERIODS)
            return pred
                    
    
    def SMA(self, dfData, dfIndices, isPredict):
        #choose whether it uses the selected period or not
        if MA_SELECT:
            pr = range(MA_PERIODS,MA_PERIODS+1)
            return self.SimpleMovingAvg(dfData, dfIndices, isPredict, pr)
        else:
            pr = range(3,6)
            return self.SimpleMovingAvg(dfData, dfIndices, isPredict, pr)
    
    def ExpoSmth(self, dfData, dfIndices, isPredict, alpha, beta):
        #inner method to calculate projection
        def rolling_ets(series, forecast_p, alpha, beta):

            pred = list()
            pred.append(float(series[0:1].values[0][0]))
            len_s = len(series)
            for n in range(1, len_s + forecast_p):
                if n==1:
                    a0 = float(series[0:1].values[0][0])
                    a1 = float(series[1:2].values[0][0])
                    (level, trend) = (a0, a1-a0)
                #forecasting period
                if n>=len_s:
                    value = pred[-1]
                #historical period
                else:
                    value = float(series[n:n+1].values[0][0])
                (last_level, level) = (level, alpha*value + 
                (1-alpha)*(level+trend))
                trend = beta*(level-last_level) + (1-beta)*trend
                pred.append(level+trend)
            return pred
        
        #train & test
        instances = len(dfData)
        
        if not isPredict:
            accList = list()
            for i in range(instances):
                trainy = dfData[i][0]
                trn = rolling_ets(trainy, 2, alpha, beta)
                trn = trn[-2:]
                trn = pd.DataFrame(trn, columns=['Vol'])
                testy = dfData[i][1]
                iacc = np.sqrt(mean_squared_error(testy,trn))
                accList.append(iacc)
            weights = [i/(instances*(instances+1)/2) for i in 
                       range(1,instances+1)]
            wacc = np.average(accList, weights=weights)
            return wacc
        else:
            trainy = dfData[0][0]
            pred = rolling_ets(trainy, FORECAST_PERIODS, alpha, beta)
            pred = pred[-FORECAST_PERIODS:]
            return pred
        
    
    def EMA(self, dfData, dfIndices, isPredict):
        
        if not isPredict:
        #train and test for the optimal alpha & beta
            global EMA_ALPHA
            global EMA_BETA
            acc = float('inf')
            for a in range(1,10,2):
                alpha = a/10
                for b in range(1,10,2):
                    beta = b/10
                    pacc = self.ExpoSmth(dfData, dfIndices, False, alpha, beta)
                    if pacc < acc:
                        acc = pacc
                        EMA_ALPHA = alpha
                        EMA_BETA = beta
            return acc
        else:
            # return prediction
            return self.ExpoSmth(dfData, dfIndices, True,EMA_ALPHA, EMA_BETA)
    
    '''
    # Parameter Optimized Holt-Winters version
    def ExpHW(self, dfData, dfIndices, isPredict):
        
        # Additive Holt-Winters method
        
        def optPara(params, *args):
            Y = args[0]
            ss = args[1]
            alpha, beta, gamma = params
            
            a = [Y[0:ss].sum()[0]/float(ss)]
            b = [Y[ss:2*ss].sum()[0]-Y[0:ss].sum()[0]/float(ss**2)]
            s = [float(Y[k:k+1]['Vol'])-a[0] for k in range(ss)]
            y = [(a[0]+b[0])+s[0]]
            
            
            for i in range(len(Y)):
                a.append(alpha*(float(Y[i:i+1]['Vol']) - s[i]) + 
                         (1-alpha)*(a[i]+b[i]))
                b.append(beta*(a[i+1]-a[i]) + (1-beta)*b[i])
                s.append(gamma*(float(Y[i:i+1]['Vol']) - (a[i]+b[i])) + 
                         (1-gamma)*s[i])
                y.append((a[i+1] + b[i+1]) + s[i+1])
            
            return np.sqrt(mean_squared_error(Y,y[:-1]))
            
        instances = len(dfData)
        if not isPredict:
            global SEASONAL
            
            acc = float('inf')
            #Loop over each season
            sea = [3,4,6,9,12]
            for ss in sea:
                invalid = 0
                flag = False
                accList = list()
                for i in range(instances):
                    trainy = dfData[i][0]
                    Y = [float(trainy[m:m+1]['Vol']) for m in 
                         range(len(trainy))]
                    # Check if twice the number of seasonal length does not 
                    # exceed the length of training data.
                    if ss > len(trainy):
                        invalid += 1
                        if invalid > instances//2:
                            flag = True
                            break
                        continue
                    testy = dfData[i][1]
                    init_values = [0.1, 0.1, 0.1]
                    boundaries = [(0,1), (0,1), (0,1)]
                    parameters = opt.fmin_l_bfgs_b(optPara, x0 = init_values,
                                               args = (trainy, ss), bounds = 
                                               boundaries, approx_grad = True)
                    alpha, beta, gamma = parameters[0]
                    # domains limit to [0,1]
                    alpha = min(1,max(0,alpha))
                    beta = min(1,max(0,beta))
                    gamma = min(1,max(0,gamma))
                    
                    
                    a = [sum(Y[0:ss])/float(ss)]
                    b = [(sum(Y[ss:ss*2]) - sum(Y[0:ss]))/ float(ss**2)]
                    s = [Y[k] - a[0] for k in range(ss)]
                    y = [(a[0]+b[0]) + s[0]]
                    
                    for j in range(len(Y)+2):
                        if j==len(Y):
                            Y.append((a[-1] + b[-1]) + s[-ss])
                        
                        a.append(alpha * (Y[j] - s[j]) + (1 - alpha) * 
                                 (a[j] + b[j]))
                        b.append(beta * (a[j + 1] - a[j]) + (1 - beta) * b[j])
                        s.append(gamma * (Y[j] - (a[j] + b[j])) + 
                                 (1 - gamma) * s[j])
                        y.append((a[j + 1] + b[j + 1]) + s[j + 1])
                    
                    train = pd.DataFrame(Y[-2:], columns=['Vol'])
                    iacc = np.sqrt(mean_squared_error(testy,train))
                    accList.append(iacc)
                    
                
                #if season exceeds the training, break the outer loop.
                if flag:
                    break
                
                valid = instances - invalid
                weights = [i/(valid*(valid+1)/2) for i in 
                       range(1,valid+1)]
                wacc = np.average(accList, weights=weights)
                
                #compare rmse to other seasonal periods
                if wacc < acc:
                    acc = wacc
                    SEASONAL = ss
                
            return acc
        else:
            trainy = dfData[0][0]
            Y = [float(trainy[m:m+1]['Vol']) for m in range(len(trainy))]
            init_values = [0.1, 0.1, 0.1]
            boundaries = [(0,1), (0,1), (0,1)]
            parameters = opt.fmin_l_bfgs_b(optPara, x0 = init_values,args = 
                                           (trainy, SEASONAL), bounds = 
                                           boundaries, approx_grad = True)
            alpha, beta, gamma = parameters[0]
            # domains limit to [0,1]
            alpha = min(1,max(0,alpha))
            beta = min(1,max(0,beta))
            gamma = min(1,max(0,gamma))
            
            #initialize level, trend, seasonal, forecast
            a = [sum(Y[0:SEASONAL])/float(SEASONAL)]
            b = [(sum(Y[SEASONAL:SEASONAL*2]) - sum(Y[0:SEASONAL]))/ 
                 float(SEASONAL**2)]
            s = [Y[k] - a[0] for k in range(SEASONAL)]
            y = [(a[0]+b[0]) + s[0]]
            
            for j in range(len(Y)+FORECAST_PERIODS):
                if j==len(Y):
                    Y.append((a[-1] + b[-1]) + s[-SEASONAL])
                
                a.append(alpha * (Y[j] - s[j]) + (1 - alpha) * (a[j] + b[j]))
                b.append(beta * (a[j + 1] - a[j]) + (1 - beta) * b[j])
                s.append(gamma * (Y[j] - (a[j] + b[j])) + (1 - gamma) * s[j])
                y.append((a[j + 1] + b[j + 1]) + s[j + 1])
            
            print('alpha:',alpha, 'beta:',beta,'gamma:',gamma)
            return Y[-FORECAST_PERIODS:]
    '''
    
    # Enumerated Holt-Winters version
    def ExpHW1(self, dfData, dfIndices, isPredict):
        
        # Additive Holt-Winters method
            
        instances = len(dfData)
        if not isPredict:
            instances = len(dfData)
            global SEASONAL
            global HW_ALPHA
            global HW_BETA
            global HW_GAMMA
            
            acc = float('inf')
            #Loop over each parameter
            sea = [3,4,6,9,12]
            alList = [0.1,0.3,0.5,0.7,0.9]
            beList = [0.1,0.3,0.5,0.7,0.9]
            gaList = [0.1,0.3,0.5,0.7,0.9]
            
            accList = list()
            for ss in sea:
                for al in alList:
                    for be in beList:
                        for ga in gaList:
                            invalid = 0
                            flag = False
                            accList = list()
                            for i in range(instances):
                                trainy = dfData[i][0]
                                Y = [float(trainy[m:m+1]['Vol']) for m in 
                                     range(len(trainy))]
                                # Check if twice the number of seasonal length does not 
                                # exceed the length of training data.
                                if ss > len(trainy):
                                    invalid += 1
                                    if invalid > instances//2:
                                        flag = True
                                        break
                                    continue
                                testy = dfData[i][1]
                                alpha = al
                                beta = be
                                gamma = ga
                                
                                
                                a = [sum(Y[0:ss])/float(ss)]
                                b = [(sum(Y[ss:ss*2]) - sum(Y[0:ss]))/ float(ss**2)]
                                s = [Y[k] - a[0] for k in range(ss)]
                                y = [(a[0]+b[0]) + s[0]]
                                
                                for j in range(len(Y)+2):
                                    if j==len(Y):
                                        Y.append((a[-1] + b[-1]) + s[-ss])
                                    
                                    a.append(alpha * (Y[j] - s[j]) + (1 - alpha) * 
                                             (a[j] + b[j]))
                                    b.append(beta * (a[j + 1] - a[j]) + (1 - beta) * b[j])
                                    s.append(gamma * (Y[j] - (a[j] + b[j])) + 
                                             (1 - gamma) * s[j])
                                    y.append((a[j + 1] + b[j + 1]) + s[j + 1])
                                
                                train = pd.DataFrame(Y[-2:], columns=['Vol'])
                                iacc = np.sqrt(mean_squared_error(testy,train))
                                accList.append(iacc)
                                
                            
                            #if season exceeds the training, break the outer loop.
                            if flag:
                                break
                            
                            valid = instances - invalid
                            weights = [i/(valid*(valid+1)/2) for i in 
                                   range(1,valid+1)]
                            wacc = np.average(accList, weights=weights)
                            
                            #compare rmse to other seasonal periods
                            if wacc < acc:
                                acc = wacc
                                HW_ALPHA = al
                                HW_BETA = be
                                HW_GAMMA = ga
                                SEASONAL = ss
                
            return acc
        else:
            trainy = dfData[0][0]
            Y = [float(trainy[m:m+1]['Vol']) for m in range(len(trainy))]
            
            alpha = HW_ALPHA
            beta = HW_BETA
            gamma = HW_GAMMA
            
            #initialize level, trend, seasonal, forecast
            a = [sum(Y[0:SEASONAL])/float(SEASONAL)]
            b = [(sum(Y[SEASONAL:SEASONAL*2]) - sum(Y[0:SEASONAL]))/ 
                 float(SEASONAL**2)]
            s = [Y[k] - a[0] for k in range(SEASONAL)]
            y = [(a[0]+b[0]) + s[0]]
            
            for j in range(len(Y)+FORECAST_PERIODS):
                if j==len(Y):
                    Y.append((a[-1] + b[-1]) + s[-SEASONAL])
                
                a.append(alpha * (Y[j] - s[j]) + (1 - alpha) * (a[j] + b[j]))
                b.append(beta * (a[j + 1] - a[j]) + (1 - beta) * b[j])
                s.append(gamma * (Y[j] - (a[j] + b[j])) + (1 - gamma) * s[j])
                y.append((a[j + 1] + b[j + 1]) + s[j + 1])
            
            print('alpha:',alpha, 'beta:',beta,'gamma:',gamma)
            return Y[-FORECAST_PERIODS:]
    
    
    def ARIMA(self, dfData, dfIndices, isPredict):
        #Turn off warning
        warnings.simplefilter('ignore')
        
        if not isPredict:
            domain = range(4)
            instances = len(dfData)
            global ARIMA_P
            global ARIMA_D
            global ARIMA_Q
            acc = float('inf')
            for p in domain:
                for d in domain:
                    for q in domain:
                        order = (p,d,q)
                        accList = list()
                        invalid = 0
                        flag = False
                        for i in range(instances):
                            trainy = dfData[i][0]
                            #Insufficient degree of freedom
                            if len(trainy) <= (p+d+q+1):
                                invalid += 1
                                if invalid > instances//2:
                                    break
                                continue
                            try:
                                model = ar(trainy, order)
                                model = model.fit()
                                pred = model.forecast(2)
                            except ValueError:
                                invalid += 1
                                if invalid > instances//2:
                                    break
                                continue
                            except:
                                flag = True
                                break
                            
                            pred = pd.DataFrame(pred[0], columns=['Vol'])
                            testy = dfData[i][1]
                            try:
                                iacc = np.sqrt(mean_squared_error(testy,pred))
                            except ValueError:
                                flag = True
                                break
                            accList.append(iacc)
                        
                        if invalid > instances//2 or flag:
                            continue
                        valid = instances - invalid
                        weights = [i/(valid*(valid+1)/2) for i in 
                                   range(1,valid+1)]
                        wacc = np.average(accList, weights=weights)
                        if wacc < acc:
                            acc = wacc
                            ARIMA_P = p
                            ARIMA_D = d
                            ARIMA_Q = q
            return acc
        
        else:
            trainy = dfData[0][0]
            order = (ARIMA_P, ARIMA_D, ARIMA_Q)
            model = ar(trainy, order)
            model = model.fit()
            pred = model.forecast(FORECAST_PERIODS)
            return list(pred[0])
    
    
    def CrossValidate(self, data):

        leng = (len(data)//2)-1
        tscv = TimeSeriesSplit(n_splits=leng)
        ListOfTuples = list()
        ListOfIndices = list()
        
        #train & test and find the most suitable model
        for train, test in tscv.split(data):
            tr = data[train[0]:train[-1]+1]
            ts = data[test[0]:test[-1]+1]
            ListOfTuples.append((tr,ts))
            ListOfIndices.append((train, test))
        
        AccScore = list()
        
        #Linear Regression - 0
        accLinReg = self.LinearReg(ListOfTuples, ListOfIndices, False)
        AccScore.append(accLinReg)
        print("Linear Reg:",accLinReg)
        
        #Curve-fitting - 1
        accCurvFit = self.CurveFitting(ListOfTuples, ListOfIndices, False)
        AccScore.append(accCurvFit)
        print("Curve-fitting("+str(CF_DEGREE)+"):", accCurvFit)
        
        #Simple Moving Average - 2
        accSMA = self.SMA(ListOfTuples, ListOfIndices, False)
        AccScore.append(accSMA)
        print("SMA("+str(MA_PERIODS)+"):",accSMA)
        
        #Exponential Smoothing - 3
        accEMA = self.EMA(ListOfTuples, ListOfIndices, False)
        AccScore.append(accEMA)
        print("EMA("+str(EMA_ALPHA)+","+str(EMA_BETA)+"):",accEMA)

        #Holt-Winters Method - 4
        accHW = self.ExpHW1(ListOfTuples, ListOfIndices, False)
        AccScore.append(accHW)
        print("Holt-Winters("+str(HW_ALPHA)+","+str(HW_BETA)+","+str(HW_GAMMA)+
                            ","+str(SEASONAL)+"):",accHW)
        '''
        for a in range(5):
            AccScore.append(1000000)
        '''
        #ARIMA - 5
        accARIMA = self.ARIMA(ListOfTuples, ListOfIndices, False)
        AccScore.append(accARIMA)
        print("ARIMA ("+str(ARIMA_P)+","+str(ARIMA_D)+","+str(ARIMA_Q)+"):",
              accARIMA)
        
        
        #Select the best model
        minerr = min(AccScore)
        indMinErr = AccScore.index(minerr)
        modelname = self.ModelNames[indMinErr]
        
        #Forecast data using the chosen model (least error)
        buildIndices = [i for i in range(1,len(data)+1)]
        predIndices = [j for j in range(len(data)+1,len(data)+1+FORECAST_PERIODS)]
        
        model = None
        
        while model == None:
            try:
                model = self.MODELS[indMinErr]([(data,None)], 
                                        [(buildIndices,predIndices)], True)
            except:
                AccScore[indMinErr] = float('inf')
                minerr = min(AccScore)
                indMinErr = AccScore.index(minerr)
                modelname = self.ModelNames[indMinErr]
                model = self.MODELS[indMinErr]([(data,None)], 
                                        [(buildIndices,predIndices)], True)
            
            check = all(m >= 0 for m in model)
            #check if all predicted period are not less than 0
            if check:
                break
            elif minerr == float('inf'):
                return None
            else:
                AccScore[indMinErr] = float('inf')
                minerr = min(AccScore)
                indMinErr = AccScore.index(minerr)
                modelname = self.ModelNames[indMinErr]
                model = None
        
        print(data)
        start = self.Max_Date + relativedelta(months=+1)
        end = self.Max_Date + relativedelta(months=+FORECAST_PERIODS)
        dtRange = pd.date_range(start=start, end=end, freq='MS')
        dtRange = dtRange.strftime('%Y-%m')
        pred = pd.DataFrame(model, columns=['Vol'], index = dtRange)
        print(pred)
        return (pred, modelname)
        
    
    #main method to forecast...
    def forecast(self):
        data_dict = self.summarize_data()
        
        all_skus = len(data_dict)
        ind = 0
        
        if self.app != None:
            self.app.updateProgress(all_skus,ind)
        
        #iterate over SKUs
        for sku in data_dict.keys():
            
            #Measure time
            start_time = time.time()
            
            output = self.CrossValidate(data_dict[sku])
            # if program cannot find a proper model, skip this sku
            if output == None:
                all_skus -= 1
                continue
            
            (model, name) = output
            model = model.transpose()
            model = model.rename({'Vol':ind})
            
            
            if ind == 0:
                results = pd.DataFrame({'Element': [sku], 'Prev(-2)':
                    [data_dict[sku][-2:-1]['Vol'][0]], 'Prev(-1)':
                        [data_dict[sku][-1:]['Vol'][0]], 'Model':[name]},
                        index = [ind])
                results = results[['Element','Prev(-2)','Prev(-1)','Model']]
                results = pd.concat([results, model], axis=1, join='inner')
            else:
                newRow = pd.DataFrame({'Element': [sku], 'Prev(-2)':
                    [data_dict[sku][-2:-1]['Vol'][0]], 'Prev(-1)':
                        [data_dict[sku][-1:]['Vol'][0]], 'Model':[name]},
                        index = [ind])
                newRow = newRow[['Element','Prev(-2)','Prev(-1)','Model']]
                newRow = pd.concat([newRow, model], axis=1, join='inner')
                results = pd.concat([results, newRow])
            
            #show time
            end_time = time.time()
            sku_time = end_time - start_time
            print("Time (sec):",str(sku_time))
            
            ind += 1
            if self.app != None:
                
                self.app.updateProgress(all_skus,ind)
            
        return results
    
    def __init__(self, df_raw, app=None):
        self.app = app
        self.df_raw = pd.DataFrame(df_raw)
        #Master of Models
        self.MODELS = {0:self.LinearReg, 1:self.CurveFitting, 2:self.SMA,
                       3:self.EMA, 4:self.ExpHW1, 5:self.ARIMA}
        self.ModelNames = {0:'Linear Regression', 1:'Curve Fitting', 
                           2:'Simple Moving Average', 3:'Exponential Smoothing',
                           4:'Holt-Winters Method', 5:'ARIMA'}


#UI class
class Application(tk.Frame):
    
    #browse input Excel file
    def browse_file(self):
        self.df=None
        self.runButton.configure(state=tk.DISABLED)
        self.exportButton.configure(state=tk.DISABLED)
        self.filename = filedialog.askopenfilename()
        filelist = self.filename.split('.')
        if self.filename == '':
            return
        elif filelist[-1] not in ('xlsx','xls'):
            messagebox.showwarning('Warning!',
                'The selected file is not an Excel file.')
            return
        file = pd.ExcelFile(self.filename)
        try:
            df = file.parse('Forecast_Input')
            if any(VALID_Columns!=df.columns.values):
                raise TypeError
        except TypeError:
            messagebox.showwarning('Warning!',
                'This Excel is in the wrong format.')
            return
        except:
            messagebox.showwarning('Warning!',
                'This Excel does not contain valid worksheeet.')
            return
        finally:
            file.close()
        messagebox.showinfo(message='Loaded Completed!')
        self.df=df
        self.runButton.configure(state=tk.NORMAL)
        
    #return a copy of dataframe object
    def get_df(self):
        try:
            return pd.DataFrame(self.df).copy()
        except:
            raise Exception("No dataframe!")
        
    
    def stopThread(self):
        self.ev.set()
        self.thread.join(0)
        self.quit()
    
    #run forecast logic from TSA class
    def run(self):
        self.fileButton.configure(state=tk.DISABLED)
        self.runButton.configure(state=tk.DISABLED)
        self.exportButton.configure(state=tk.DISABLED)
        self.stopButton.configure(state=tk.NORMAL)
        
        #disable Close button
        self.isForecast = True
        
        #Set Forecast Period
        global FORECAST_PERIODS
        FORECAST_PERIODS = self.dropperiod.getvar("Var")
        
        df_raw = self.get_df()
        self.tsa = TSA(df_raw, self)
        self.predictions = self.tsa.forecast()
        messagebox.showinfo(message='Forecast Completed!')
        
        self.exportButton.configure(state=tk.NORMAL)
        self.fileButton.configure(state=tk.NORMAL)
        self.runButton.configure(state=tk.NORMAL)
        self.stopButton.configure(state=tk.DISABLED)
        
        #Enable Close button
        self.isForecast = False
        
    def execute(self):
        try:
            self.ev = th.Event()
            self.thread = th.Thread(target = self.run)
            self.thread.setDaemon(True)
            self.thread.start()
        except:
            print('Thread Error')
    
    def updateProgress(self, totalSKU, fshSKU):
        pct = round(fshSKU/totalSKU*100,2)
        prog = 'Progress: '+str(fshSKU)+'/'+str(totalSKU)+' ('+str(pct)+'%)'
        self.progLabel.configure(text = prog)
        
    def export_file(self):
        if self.predictions is None:
            messagebox.showwarning('Warning!',
                'Export file error!')
            return
        
        filename = filedialog.asksaveasfilename(defaultextension='.xlsx')
        if filename == '':
            return
        writer = pd.ExcelWriter(filename)
        self.predictions.to_excel(excel_writer = writer, sheet_name = 
                                  'Forecast_Output', index = False)
        messagebox.showinfo(message='Export to Excel Completed!')
        writer.save()
    
    #Exit window    
    def close_window(self):
        if not self.isForecast:
            self.quit()
    
    #create layout
    def createWidget(self, master):
        self.master = master
        
        #create browse button
        self.fileButton = tk.Button(master)
        self.fileButton["text"] = "Browse..."
        self.fileButton["command"] = self.browse_file
        
        #self.fileButton.pack(side=RIGHT)
        #self.fileButton.place(relx=0.5, rely=0.5)
        
        #create run button
        self.runButton = tk.Button(master, state=tk.DISABLED)
        self.runButton["text"]="Forecast"
        self.runButton["command"] = self.execute
        
        #create export button
        self.exportButton = tk.Button(master, state=tk.DISABLED)
        self.exportButton["text"]="Export..."
        self.exportButton["command"]=self.export_file
        
        self.stopButton = tk.Button(master, state=tk.DISABLED)
        self.stopButton["text"] = "Stop"
        self.stopButton["command"] = self.stopThread
        
        #Forecast Label
        self.FCLabel = tk.Label(master, text="# Projection:", 
                                font=("Helvetica", 9))
        
        #Progress Label
        txt = "Progress: N/A"
        self.progLabel = tk.Label(master, text = txt, font=("Helvetica", 9))
        
        #list variables
        variable = tk.StringVar(master=master, name="Var")
        variable.set(2) #default
        
        FC_period = range(1,19)
        
        self.dropperiod = tk.OptionMenu(master, variable, *FC_period)
        
        #set location of buttons
        self.fileButton.grid(column=0,row=0,padx=10,pady=5)
        self.runButton.grid(column=1,row=0,padx=10,pady=5)
        self.exportButton.grid(column=2,row=0,padx=10,pady=5)
        self.stopButton.grid(column=3,row=0,padx=10,pady=5)
        self.FCLabel.grid(column=0,row=1,padx=0,pady=5)
        self.dropperiod.grid(column=1,row=1, padx=10, pady=5)
        self.progLabel.grid(column=2,row=1, padx=10, pady=5)
        

    
    def __init__(self, master=None):
        master.minsize(400,300)
        master.resizable(width=False, height= False)
        master.protocol('WM_DELETE_WINDOW', self.close_window)
        tk.Frame.__init__(self,master)
        self.predictions = None
        
        self.isForecast = False
        #self.pack()
        self.createWidget(master)


def main():
    
    parent = tk.Tk()
    parent.wm_title('Forecast Platform')
    app = Application(parent)
    app.mainloop()
    
    try:
        parent.destroy()
    except tk.TclError:
        pass

'''test module'''
def test():
    filename = 'C:/Users/suradiss/Desktop/TPC Forecast/raw data PPPC plant.xlsx'
    file = pd.ExcelFile(filename)
    df = file.parse('Forecast_Input')
    df = pd.DataFrame(df)
    tsa = TSA(df)
    tsa.forecast()


if __name__=="__main__":
    main()
    #test()