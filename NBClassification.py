# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:50:06 2019

@author: MAMEN
"""

import pandas as pd
import numpy as np

class NBClassification:
    
    X_train = pd.DataFrame()
    Y_train = []
    X_test = pd.DataFrame()
    Y_test = []
    neighbours = []
    res = {'actual': [], 'predicted':[]}
    p_yes = 0
    p_no = 0
    train = pd.DataFrame()
    test = pd.DataFrame()
    
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        self.train = self.X_train
        self.train['class'] = self.Y_train
        
        self.test = self.X_test
        self.test['class'] = self.Y_test
        
        
        self.p_yes = self.train[(self.train['class']==1)].shape[0]/len(self.train)
        self.p_no = self.train[(self.train['class']==0)].shape[0]/len(self.train)
        
    def __pdf(self,x,mean,sigma):
        return (1/(np.sqrt(2*np.pi)*sigma)) * np.exp(-1*(x-mean)**2/(2*sigma**2))
    
    
    def predict(self):
        probability_yes = []
        probability_no = []
        y_pred = []
        for i,row in self.test.iterrows():
            for c in self.test.columns[:-1]:                
                #edible                
                data_c_yes = self.train[(self.train[c]==row[c]) & (self.train['class']==1)]                
                data_yes = self.train[(self.train['class']==1)]
                #mean_yes = data_yes[c].mean()
                #var_yes = data_yes[c].var()             
                #likelihood_yes = self.__pdf(self.test[c][i], mean_yes, np.sqrt(var_yes))
                probability_yes.append(data_c_yes.shape[0]/data_yes.shape[0])
                
                
                #poison
                data_c_no = self.train[(self.train[c]==row[c]) & (self.train['class']==0)]
                data_no = self.train[(self.train['class']==0)]
                #mean_no = data_no[c].mean()
                #var_no = data_no[c].var()
                #likelihood_no = self.__pdf(self.test[c][i], mean_no, np.sqrt(var_no))
                probability_no.append(data_c_no.shape[0]/data_no.shape[0])
            
         
            p1 = np.prod(probability_yes) * self.p_yes
            p0 = np.prod(probability_no) * self.p_no
            pred = 1*(p1>p0)
            y_pred.append(pred)
            probability_yes = []
            probability_no = []
        
        self.res['predicted'] = np.array(y_pred)
        self.res['actual'] = self.Y_test
        return self.res
        
       