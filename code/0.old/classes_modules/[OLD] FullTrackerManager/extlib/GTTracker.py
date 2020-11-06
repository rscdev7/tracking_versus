"""
@author: rscalia
@date: Fri 17/04/2020

Questo componente serve per estrarre le informazioni di Ground Truth dal TV Dataset.

"""

import os
import sys


class GTTracker (object):


    def __init__ (self, pDataPath, pThreshold = None, pLimit = None, pSystem = "Inc"):

        self._dataPath   = pDataPath
        self._data       = {}
        self._threshold  = pThreshold
        self._limit      = pLimit
        self._system     = pSystem

    def compute (self):
        f = open(self._dataPath, 'r')
        lines = f.readlines()
        f.close()


        for line in lines:
            new_line =line.split(',')[:7]
            
            #Prelevo la confidenza della predizione
            cf = float(new_line[6])

            #Verifico confidenza sfruttando una soglia
            if (self._threshold == None or cf >= self._threshold):
                new_line = list ( map(lambda x: int(float(x)), new_line) )
                new_line[6] =cf

                frame_number = new_line[0]

                result = frame_number in self._data.keys()

                #L'angolo in alto a destra delle immagini è considerato (1,1) nel dataset
                X = new_line[2] - 1
                Y = new_line[3] - 1
                W = new_line[4] 
                H = new_line[5]
                track_id = new_line[1]

                if (self._system   == "Inc"):
                    bbox = [ X, Y, X + W, Y + H, track_id ]
                elif (self._system == "WH"):
                    bbox = [ X, Y, W, H, track_id ]

                if (result == False):
                    self._data[frame_number] = []
                    
                    
                self._data[frame_number].append(bbox)

        
        if (self._limit != None):

            for f_number in self._data.keys():
                limit = self._limit
                self._data[f_number] = self._data[f_number][:limit]
