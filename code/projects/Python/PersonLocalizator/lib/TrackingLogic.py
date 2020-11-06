"""
@author    :     rscalia
@date      :     Tue 04/08/2020

Questo componente implementa la logica di Tracking all'interno dell'applicativo PersonLocalizer.

"""


import cv2
import sys
sys.path.append('./extlib')
from scipy.spatial import distance

from DataManager import *
from ManagedCentroid import *
from ManagedCSRT import *
from ManagedSORT import *
from LKTracker   import *
from BoxPrinter import *
import random


NOT_MATCHED = -1

class TrackingLogic (object):

    def __init__ (self):

        #Parametri Data Manager
        self._threshold                 = 1.0
        self._limit                     = None
        self._system                    = "Inc"

        #Parametri Lucas-Kanade
        self._lkParams                  = dict( winSize  = (15,15),
                                                maxLevel = 2,
                                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self._featuresTrkParams         = dict( maxCorners = 100,
                                                qualityLevel = 0.3,
                                                minDistance = 7,
                                                blockSize = 7 )

        #Parametri BoxPrinter
        self._paletteWidth              = 20
        self._thicknessLine             = 4


        #Parametri Logica Tracking
        self._dataManager               = None
        self._trackers                  = []
        self._printer                   = None

        self._idLKEnumerator            = 1
        self._idCSRTEnumerator          = 1

        self._sortActivated             = None
        self._centroidActivated         = None

        self._printStack                = {}


    def init (self,pDataPath):
        self._dataManager               = DataManager (pDataPath,self._threshold,self._limit,self._system)
        self._dataManager.readData()

        self._trackers                  = []
        self._printer                   = BoxPrinter(self._paletteWidth, self._thicknessLine)

        self._idLKEnumerator            = 1
        self._idCSRTEnumerator          = 1
        

        self._printStack                = {}

        self._sortActivated             = None
        self._centroidActivated         = None


    def compute (self, pImage, pFrameNumber):
        self._printStack                = {}

        to_remove                       = False
        
        for trk in self._trackers:

            if (isinstance(trk, LKTracker) == True):

                if (pFrameNumber in self._dataManager._data.keys()):
                
                    boxes = self._dataManager._data[pFrameNumber]
                    boxes = list( filter (lambda x: x[:4] , boxes) )

                    trk.compute(pImage, boxes)

                    if (type(trk) != type(None) and type(trk._currentPosition) != type(None)):
                        cp_boxes    = trk._currentPosition

                        key         = (LK_CODE, trk._idx)
                        values      = cp_boxes[key]

                        self._printStack[key]  = values
                    else:
                        to_remove = True


            elif (isinstance(trk, ManagedCSRT) == True):
                trk.computeAndStore(pImage)
                key  = trk._key

                if (type(trk) != type(None) and type(trk._trackInfo[key]) != type(None)):
                    cp_boxes = trk._trackInfo

                    x_1 = cp_boxes[key][0]
                    y_1 = cp_boxes[key][1]
                    x_2 = cp_boxes[key][0] + cp_boxes[key][2]
                    y_2 = cp_boxes[key][1] + cp_boxes[key][3]

                    self._printStack[key] = [x_1, y_1, x_2, y_2]
                else:
                    to_remove = True


            elif (isinstance(trk, ManagedSORT) == True):

                if (pFrameNumber in self._dataManager._data.keys()):

                    boxes = self._dataManager._data[pFrameNumber]
                    boxes = list ( map ( lambda x: x[:4], boxes  ) )

                    trk.computeAndStore(boxes)

                    cp_boxes = trk._trackInfo

                    for key, value in zip(cp_boxes.keys(), cp_boxes.values()):
                        self._printStack[key] = value


            elif (isinstance(trk, ManagedCentroid) == True):

                if (pFrameNumber in self._dataManager._data.keys()):

                    boxes = self._dataManager._data[pFrameNumber]
                    boxes = list ( map ( lambda x: x[:4], boxes  ) )

                    trk.computeAndStore(boxes)

                    cp_boxes = trk._trackInfo

                    for key, value in zip(cp_boxes.keys(), cp_boxes.values()):
                        self._printStack[key] = value

            
        self._printer.compute(pImage, self._printStack)

        

    def registerTracker (self, pAlgorithm, pROI, pFrameNumber, pImage = None):


        if (pAlgorithm == "Lucas-Kanade Tracker"):

            if (pFrameNumber in self._dataManager._data.keys()):
                boxes = self._dataManager._data[pFrameNumber]
                res  = self._findMostSimBox(boxes,pROI)

                if (res != NOT_MATCHED):
                    pROI = boxes[res]
                    
                    trk_engine    = LKTracker(pImage, pROI, self._idLKEnumerator, None, self._lkParams, self._featuresTrkParams)

                    self._trackers.append(trk_engine)

                    self._idLKEnumerator += 1
                else:
                    print ("\n NOT MATCHED \n")


        if (pAlgorithm == "Centroid Tracker"):

            if (pFrameNumber in self._dataManager._data.keys()):
                
                boxes           = self._dataManager._data[pFrameNumber]
                boxes           = list ( map ( lambda x: x[:4], boxes  ) )

                if (self._centroidActivated == None):
                    trk_engine      = ManagedCentroid ()
                    trk_engine.computeAndStore(boxes)

                    self._trackers.append(trk_engine)
                    self._centroidActivated  = self._trackers.index(trk_engine)


        if (pAlgorithm == "CSRT"):

            pROI = [ pROI[0], pROI[1], pROI[2] - pROI[0], pROI[3] - pROI[1] ]
            trk_engine = ManagedCSRT(pImage, pROI, self._idCSRTEnumerator)

            self._trackers.append(trk_engine)
            self._idCSRTEnumerator += 1


        if (pAlgorithm == "SORT"):
            
            if (pFrameNumber in self._dataManager._data.keys()):
                boxes           = self._dataManager._data[pFrameNumber]
                boxes           = list ( map ( lambda x: x[:4], boxes  ) )

                if (self._sortActivated == None):
                    trk_engine      = ManagedSORT ()
                    trk_engine.computeAndStore(boxes)

                    self._trackers.append(trk_engine)
                    self._sortActivated  = self._trackers.index(trk_engine)

    
    def _findMostSimBox (self, pBoxes, pROI):
        vote                    = []

        for idx,box in enumerate(pBoxes):
            
            overlap             = self._jaccard(pROI, box)

            #CALCOLO CENTROIDI
            p1_x                =  box[0] + box[2]//2
            p1_y                =  box[1] + box[3]//2
            p1                  = [p1_x,p1_y]

            p2_x                =  pROI[0] + pROI[2]//2
            p2_y                =  pROI[1] + pROI[3]//2
            p2                  =  [p2_x,p2_y]

            dist                = distance.euclidean(p1,p2)

            if (overlap >= 0.5):
                overlap         = 1

            if (overlap >= 0.7):
                overlap         = 2
            
            if (overlap >= 0.9):
                overlap         = 3

            rec                 = (idx,overlap, dist)

            vote.append(rec)
        

        exited = False
        for i in reversed( range(1,4) ):

            restricted_vote     = list( filter (lambda x: x[1] == i, vote) )
            if (len (restricted_vote) > 0):
                exited          = True
                break
        
        if (exited == False):
            return NOT_MATCHED
        else: 
            restricted_vote     = list( map (lambda x: (x[0],x[2]) , restricted_vote) )
            final_vote          = sorted(restricted_vote, key=lambda x: x[1])
            verdict             = final_vote[0][0]

            return verdict


    def _jaccard (self,pBoxA, pBoxB):

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
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

        
