# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:44:56 2019

@author: MAMEN
"""

import pandas as pd
import numpy as np
import math
import operator

class KNNClassification:
    K=0
    X_train = pd.DataFrame()
    Y_train = []
    X_test = pd.DataFrame()
    Y_test = []
    neighbours = []
    predicted = []
    res = { 'actual': [], 'predicted': [] }
    
    def __init__(self, K, X_train, Y_train, X_test, Y_test):
        self.K = K
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        self.X_train['class'] = self.Y_train
    
    def __getDistance(self,test,train):
        distance = ((test-train)**2).sum()
        return math.sqrt(distance)
    
    def __getNeighbour(self, test):
        distances = []       
        self.X_train[:-1].apply(lambda row: distances.append((self.__getDistance(test,row),row['class'])), axis=1)
        
        #for i,row in self.X_train.iterrows():
        #    calcDistance = self.__getDistance(test, row)
        #    distances.append((row, calcDistance, self.Y_train[i]))
        
        distances.sort(key = operator.itemgetter(0))
        
        self.neighbours = distances[:self.K]
        
    def __voteClass(self):
        c = { "e": 0, "p":0 }
        for i in range(self.K):
            if self.neighbours[i][1] == 1:
                c['e'] += 1
            else:
                c['p'] += 1
                
        if c['e'] > c['p']:
            self.neighbours = []
            return 1
        else:
            self.neighbours = []
            return 0
            
        
        
    def predict(self):
        #for i,row in self.X_test.iterrows():
        #    if index == 10 : break        
        #    self.__getNeighbour(row)
        #    y_pred = self.__voteClass()
        #    predicted.append(y_pred)
        #    print(y_pred)
        
        self.X_test.apply(self.__doPred, axis=1)
        self.res['predicted'] = np.array(self.predicted)
        self.res['actual'] = self.Y_test
        
        return self.res
    
    def __doPred(self,row):
        self.__getNeighbour(row)
        y_pred = self.__voteClass()
        self.predicted.append(y_pred)

            
            