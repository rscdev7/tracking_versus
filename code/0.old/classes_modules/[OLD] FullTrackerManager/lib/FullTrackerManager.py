"""
@author: rscalia
@date: Fri 03/07/2020

Questo componente serve per .........

"""

import sys
sys.path.append ('../extlib')
from scipy.optimize import linear_sum_assignment

import os
import cv2
import numpy as np
from ManagedTracker import *


CSTR_CODE       = "CSTR"
CENTORID_CODE   = "Centroid"
MIL_CODE        = "MIL"
SORT_CODE       = "SORT"

ARGS_ERROR      = -1
START_ID        = 1

class FullTrackerManager (object):

    def __init__ (self, pChoosenAlgorithm, pJaccardThr):

        self._algorithm         = pChoosenAlgorithm

        if (self._algorithm == CSTR_CODE or self._algorithm == MIL_CODE):
            self._trackers = cv2.MultiTracker_create()
        else:
            self._trackers          = []
        self._trackInfo         = {}
        self._idEnumerator      = START_ID
        self._jacThr            = pJaccardThr


    def compute (self, pImage = None, pBoxes = None):

        if (type(pImage) == type(None) and type(pBoxes) == type(None)):
            print ("\n!!! Errore nei parametri passati \n")
            return ARGS_ERROR


        if (self._algorithm == CSTR_CODE or self._algorithm == MIL_CODE):
            self._stCompute(pImage,pBoxes)
        else:
            self._mtCompute(pBoxes)
            

    def _stCompute (self, pImage, pBoxes):
        
      
        if (type(self._trackers) == tuple):
            for box in pBoxes:

                tracker = None
                if (self._algorithm == MIL_CODE):
                    tracker = cv2.TrackerMIL_create()
                else:
                    tracker = cv2.TrackerCSRT_create()
                
                self._trackers.add (tracker, pImage, tuple(box))

                key  = (self._algorithm, self._idEnumerator)
                self._idEnumerato += 1

                self._trackInfo[key] = [ box[0] , box[1] , box[2] , box[3] ]

        else:

            #Aggiornamento Tracker Attuali
            (success, boxes) = list(self._trackers.update(pImage))

            if (success == True):
                self._trackInfo = {}

                for idx,box in enumerate(boxes):
                    key = (self._algorithm, idx + 1)
                    self._trackInfo[key] = box

                #EVENTUALE CREAZIONE NUOVI TRACKER
                distances = np.ones( ( len(pBoxes) , len(boxes) ) )
                for i,box_s in enumerate(pBoxes):
                    for j,box_d in enumerate(boxes):
                        distances[i][j] = self._jaccard(box_s,box_d)

                if (self._jacThr != None):
                    distances = (distances >= self._jacThr) * distances

                row_ind, row_col = linear_sum_assignment(distances , maximize=True)

                box_enumeration = set(range(len(pBoxes)))
                new_boxes = set(box_enumeration).difference(row_ind)

                if (list(new_boxes) != []):
                    for i in new_boxes:

                        tracker = None
                        if (self._algorithm == MIL_CODE):
                            tracker = cv2.TrackerMIL_create()
                        else:
                            tracker = cv2.TrackerCSRT_create()
                        
                        self._trackers.add (tracker, pImage, tuple(pBoxes[i]))

                        key  = (self._algorithm, self._idEnumerator)
                        self._idEnumerator += 1

                        self._trackInfo[key] = [ pBoxes[i][0] , pBoxes[i][1] , pBoxes[i][2] , pBoxes[i][3] ]

                
                #EVENTUALE CANCELLAZIONE VECCHI TRACKER
                reversed_dictionary = {tuple(value) : key for (key, value) in self._trackInfo.items()}
                box_enumeration = set(range(len(boxes)))
                new_boxes = set(box_enumeration).difference(row_col)
                
                if (list(new_boxes) != []):
                    for i in new_boxes:
                        bx = boxes[i]
                        key = reversed_dictionary[tuple(bx)]

                        del self._trackInfo[key]
            else:
                print (success, boxes)
                

		        
    def _mtCompute (self, pBoxes):

        pkg                 = {}
        pkg['detections']   = pBoxes

        #Compute Informazioni Tracking
        if (self._trackers == []):
            self._trackers.append ( ManagedTracker( self._algorithm , pkg ) )
        else:
            self._trackers[-1].compute(pkg)


        #Store Nuove Informazioni
        new_data = self._trackers[-1]._currentData
        self._trackInfo = {}

        #Aggiornamento Informazioni di Tracking
        if (self._algorithm == SORT_CODE):

            for rec in new_data:
                x_1 = rec[0]
                y_1 = rec[1]
                x_2 = rec[2]
                y_2 = rec[3]
                idf = rec[4]
                key = (self._algorithm , idf)
                
                self._trackInfo[key] = [ x_1 , y_1, x_2, y_2 ]

        elif (self._algorithm == CENTORID_CODE):
            for key,value in zip(new_data.keys(),new_data.values()):
                x_1 = value[0]
                y_1 = value[1]
                x_2 = value[2]
                y_2 = value[3]
                idf = key
                key = (self._algorithm , idf)
                
                self._trackInfo[key] = [ x_1 , y_1, x_2, y_2 ]


    def _jaccard (self, pBoxA, pBoxB):

        #Calcolo coordinate rettangolo intersezione
        xA = max(pBoxA[0], pBoxB[0])
        yA = max(pBoxA[1], pBoxB[1])
        xB = min(pBoxA[2], pBoxB[2])
        yB = min(pBoxA[3], pBoxB[3])

        #Calcolo area rettangolo intersezione
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        #Calcolo aree delle due Bounding Box
        boxAArea = (pBoxA[2] - pBoxA[0] + 1) * (pBoxA[3] - pBoxA[1] + 1)
        boxBArea = (pBoxB[2] - pBoxB[0] + 1) * (pBoxB[3] - pBoxB[1] + 1)

        #Calcolo Jaccard
        union_area = float(boxAArea + boxBArea - interArea)

        if (union_area == 0.0):
            return 1.0
        else:
            iou = interArea / union_area

        return iou