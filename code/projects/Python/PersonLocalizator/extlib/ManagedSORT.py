"""
@author: rscalia
@date: Wed 15/07/2020

Questo componente rappresenta il Tracker SORT all'interno dell'architettura software.

"""

import sys
sys.path.append('../extlib')

import os

from ManagedTracker import *
from sort import *
import numpy as np

SORT_TOKEN = "SORT"

class ManagedSORT (ManagedTracker):

    def __init__ (self):
        super().__init__()

        self._trackEngine = Sort ()

        #DICT FORMAT: { ("ALGO",ID): [X, Y, X_MAX, Y_MAX], ..... }
        self._trackInfo   = {}

    def computeAndStore (self, pDetections):

        #COMPUTE 
        #SORT Out Data Format: [ [x1,y1,x2,y2, ID] , [x1,y1,x2,y2, ID] , .... ]
        for lis in pDetections:

            #Aggiunto Score e Classe Fittizzia
            lis.append(1.0)
            lis.append(0)

        detects = np.array(pDetections)
        
        info               = self._trackEngine.update ( detects )


        #STORE
        self._trackInfo    = {}

        for rec in info:

            idx = ( SORT_TOKEN , int ( rec[4] ) )
            x_1 = rec[0]
            y_1 = rec[1]
            x_2 = rec[2]
            y_2 = rec[3]

            bbox = [x_1, y_1, x_2, y_2]

            self._trackInfo[idx] = bbox