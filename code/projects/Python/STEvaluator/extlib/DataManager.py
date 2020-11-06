"""
@author: rscalia
@date: Fri 17/04/2020

Questo componente serve per gestire la lettura/scrittura delle informazioni relative ai target tracciati dai tracker.

"""

import os
import sys


class DataManager (object):


    def __init__ (self, pDataPath, pThreshold = None, pLimit = None, pSystem = "Inc", pWritePath=None):

        self._dataPath          = pDataPath
        self._data              = {}
        self._threshold         = pThreshold
        self._limit             = pLimit
        self._system            = pSystem
        self._writePath         = pWritePath
        self._avaibleTargets    = {}

    #Lettura dati di Ground-Truth
    def readData (self):
        f = open(self._dataPath, 'r')
        lines = f.readlines()
        f.close()


        for line in lines:

            #Split
            new_line =line.split(',')[:7]

            #Prelevo la confidenza della predizione
            cf = float(new_line[6])


            #Verifico confidenza sfruttando una soglia
            if (self._threshold == None or cf >= self._threshold):
                new_line    = list ( map(lambda x: int(float(x)), new_line) )
                new_line[6] = cf
                frame_number = new_line[0]

                #L'angolo in alto a destra delle immagini è considerato (1,1) nel dataset
                result = frame_number in self._data.keys()
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


        #Ordino i target di ground-truth in ordine ascendente
        for f_number in self._data.keys():

            self._data[f_number]        = sorted(self._data[f_number], key=lambda x: x[4])

            if (self._limit != None):
                limit = self._limit
                self._data[f_number]    = self._data[f_number][:limit]


    #Scrivo una detection di un Tracker su File
    def write (self, pFnumber, pIdx, pBbox, pEmplSystem):

        f_number = pFnumber
        idx      = pIdx
        bbox     = [ pBbox[0]+1 , pBbox[1]+1 , pBbox[2], pBbox[3] ]

        #Eventuale Trasformazione nel sistema WH
        if (pEmplSystem != "WH" ):

            W = bbox[2] - bbox[0]
            H = bbox[3] - bbox[1]
            bbox = [ bbox[0] , bbox[1], W , H   ]


        with open(self._writePath, 'a') as fl:

            #Building Line
            line = str(f_number) + "," + str(idx) + "," + str(bbox[0]) + "," + str(bbox[1])     \
                   + "," + str(bbox[2]) + "," + str(bbox[3]) + "," + str(1) + "," + str(-1) + "," + str(-1) + "," + str(-1) + '\n'

            fl.write(line)

    #Prelevo le informazioni di una Traiettoria di uno specifico Target
    def takeTrajectory (self, pTarget):

        if (self._data != {}):
            data = list( zip( self._data.keys() , self._data.values() ) )

            v = list( map ( lambda x: ( x[0], self._cleanList(x[1], pTarget) ), data ) )
            v = list( filter ( lambda x: x[1] != [], v ) )
            v = list( map ( lambda x: x[0],v ) )
            v = sorted(v)
            
            return v
        else:
            return None

    #Rilevata tutti i Target Disponibili
    def getAllTargets (self):
        
        unique_idx = set()
        if (self._data != {}):

            for key, values in zip(self._data.keys(),self._data.values()):
                
                for rec in values:
                    prec_len = len(unique_idx)
                    unique_idx.add(rec[4])
                    current_len = len(unique_idx)

                    if (current_len > prec_len):
                        k = (rec[4])
                        self._avaibleTargets[k] = key
            
            
    #Rimuove da una lista tutte le bounding-box che non appartengono ad uno specifico target
    def _cleanList (self, pLs, pTarget):
        res = list(filter(lambda x: x[4] == pTarget,pLs))
        return res
    
