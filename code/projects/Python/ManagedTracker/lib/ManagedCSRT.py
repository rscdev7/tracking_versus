"""
@author: rscalia
@date: Wed 15/07/2020

Questo componente rappresenta l'algoritmo CSRT all'interno dell'architettura software.

"""

import sys
sys.path.append('../extlib')

import os

from ManagedTracker import *
import numpy as np
import cv2

CSRT_TOKEN = "CSRT"

class ManagedCSRT (ManagedTracker):

    def __init__ (self, pImage, pRoi, pId):
        super().__init__()

        self._trackEngine          = cv2.TrackerCSRT_create()
        self._key                  = (CSRT_TOKEN , pId)
        #DICT FORMAT: { ("ALGO",ID): [X, Y, X_MAX, Y_MAX] }
        self._trackInfo      = {}

        self._trackEngine.init( pImage , tuple(pRoi) )
        self._trackInfo[self._key] = pRoi

    def computeAndStore (self, pImage):

        #COMPUTE AND STORE
        (success, box) = self._trackEngine.update( pImage )

        if (success == True):
            self._trackInfo[self._key] = list(box)
        else:
            self._trackInfo[self._key] = None

            