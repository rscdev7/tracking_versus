import sys
sys.path.append ('../extlib')

import os
import cv2
import numpy as np
from   sort               import                  *
from   CentroidTracker    import                  *


UNKNOWN_TRACKER_TYPE  = -1


class ManagedTracker (object):


    #pInfo { 'img', 'detections', 'ID' }
    def __init__ (self, pTrackerType, pInfo):

        self._trackerTypes              = ["SORT", 'CSTR', 'MIL', 'Centroid']
        
        self._trackEngine               = None
        self._engineInfo                = pTrackerType
        self._currentData               = None
        self._id                        = None


        #SORT
        if (self._engineInfo            == self._trackerTypes[0]):
            self._trackEngine               = Sort()
            self._currentData               = self._computeSort (pInfo)
            

        #CSTR
        if (self._engineInfo            == self._trackerTypes[1]):
            self._trackEngine               = cv2.TrackerCSRT_create()
            self._trackEngine.init( pInfo['img'] , tuple(pInfo['detections']) )

            self._id                        = pInfo['ID']
            self._currentData               = [ self._id , pInfo['detections'] ]


        #MIL
        if (self._engineInfo            == self._trackerTypes[2]):
            self._trackEngine               = cv2.TrackerMIL_create()
            self._trackEngine.init( pInfo['img'] , tuple(pInfo['detections']) )

            self._id                        = pInfo['ID']
            self._currentData               = [ self._id , pInfo['detections'] ]


        #Centroid
        if (self._engineInfo  == self._trackerTypes[3]):
            self._trackEngine               = CentroidTracker()
            self._currentData               = self._computeCentroid(pInfo)


    def compute (self, pData):
        #SORT
        if (self._engineInfo            == self._trackerTypes[0]):
            payload                     = self._computeSort (pData)
            return payload
            

        #CSTR
        if (self._engineInfo            == self._trackerTypes[1]):
            payload                     = self._computeSingleTrackers (pData)
            return payload


        #MIL
        if (self._engineInfo            == self._trackerTypes[2]):
            payload                     = self._computeSingleTrackers (pData)
            return payload


        #Centroid
        if (self._engineInfo  == self._trackerTypes[3]):
            payload                     = self._computeCentroid (pData)
            return payload

    #SORT
    def _computeSort (self, pData):
        
        #CurrentData: [ [x1,y1,x2,y2, ID] , [x1,y1,x2,y2, ID] , .... ]
        for lis in pData['detections']:

            #Aggiunto Score e Classe Fittizzia
            lis.append(1.0)
            lis.append(0)

        pData['detections'] = np.array(pData['detections'])


        self._currentData               = self._trackEngine.update ( pData['detections'] )
        
        return self._currentData


    #Centroid Tracking
    def _computeCentroid (self, pData):

        #CurrentData: [ [ID, [bbox] ] , [ID, [bbox] ], .... ]
        self._currentData               = self._trackEngine.update ( pData['detections'] )
        return self._currentData


    #MIL/CSTR   
    def _computeSingleTrackers (self,pData):
        (success, box) = self._trackEngine.update( pData['img'] )

        if (success == True):
            #CurrentData: [ ID, [box] ]
            self._currentData = [ self._id , box]
            return self._currentData
        else:
            return None
    

        
            

