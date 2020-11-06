"""
@author: rscalia
@date: Wed 15/07/2020

Questo componente rappresenta il Tracker Centroid all'interno dell'architettura software.

"""

import sys
sys.path.append('../extlib')

import os

from ManagedTracker import *
from CentroidTracker import *
import numpy as np

CENTROID_TOKEN = "Centroid"

class ManagedCentroid (ManagedTracker):

    def __init__ (self):
        super().__init__()

        self._trackEngine = CentroidTracker()

        #DICT FORMAT: { ("ALGO",ID): [X, Y, X_MAX, Y_MAX], ..... }
        self._trackInfo   = {}

    def computeAndStore (self, pDetections):

        #COMPUTE 
        #Centroid Data Format: { ID : [bbox] , .... }
        info               = self._trackEngine.update ( pDetections )


        #STORE
        self._trackInfo    = {}

        for key, values in zip(info.keys(), info.values()):

            idx = ( CENTROID_TOKEN , int ( key ) )
            x_1 = values[0]
            y_1 = values[1]
            x_2 = values[2]
            y_2 = values[3]

            bbox = [x_1, y_1, x_2, y_2]

            self._trackInfo[idx] = bbox