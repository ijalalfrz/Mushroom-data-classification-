# -*- coding: utf-8 -*-
"""
Created on Mon May 27 05:22:52 2019

@author: MAMEN
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquareTest:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        
        if self.p<alpha:
            print("Kolom ",colX, " berguna untuk prediksi p-value = ",self.p)
            return self.p
        else:
            print("Kolom ",colX, " tidak berguna untuk prediksi")

        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        p = self._print_chisquare_result(colX,alpha)
        
        return p