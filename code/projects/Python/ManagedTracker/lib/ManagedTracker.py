"""
@author: rscalia
@date: Wed 15/07/2020

Questo componente rappresenta la classe astratta necessaria ad integrare un Tracker all'interno dell'architettura software custom.

"""

import sys
import os

EMPTY_INTEGER = 1

class ManagedTracker (object):

    def __init__ (self):
        self._trackInfo   = None
        self._trackEngine = None
        
    def computeAndStore (self):
        return True