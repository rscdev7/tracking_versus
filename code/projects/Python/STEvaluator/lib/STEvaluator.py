"""
@author    :     rscalia
@date      :     Wed 29/07/2020

Questo componente serve per implementare una suite di valutazione per un Tracker Single-Object.

"""

import sys
sys.path.append('../extlib')
import os
import logging
import functools

from DataManager import *
from datetime import *


EMPTY_INTEGER = -1
EMPTY_STRING  = ""
EMPTY_FLOAT   = -1.0
EMPTY_TUPLE   = (EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_INTEGER)


class STEvaluator (object):

    def __init__ (self, pGtPath, pTrkPath, pVideoName, pTrkType, pExportPath = None):

        #Sorgente dati di Ground-Truth
        self._gtPath            = pGtPath
        self._gtDataLake        = DataManager(self._gtPath, 1.0, None, "Inc")
        self._gtDataLake.readData()

        #Sorgente dati inferita dal tracker
        self._trkPath            = pTrkPath
        self._trkDataLake        = DataManager(self._trkPath, 1.0, None, "Inc")
        self._trkDataLake.readData()


        #Metriche Primitive
        self._tp                = EMPTY_INTEGER
        self._fp                = EMPTY_INTEGER
        self._fn                = EMPTY_INTEGER
        self._nFrameTrk         = EMPTY_INTEGER
        self._ious              = []


        #Metriche Complesse
        self._precision         = EMPTY_FLOAT
        self._recall            = EMPTY_FLOAT
        self._fpr               = EMPTY_FLOAT
        self._sota              = EMPTY_FLOAT
        self._trkQuality        = EMPTY_TUPLE
        self._sotp              = EMPTY_FLOAT


        #Path per esportare un log riassuntivo delle prestazione del tracker in esame
        self._exportPath        = pExportPath
        self._buildRealExportPath()


        #Informazioni sul Tracker e sul Video Elaborato
        self._vidName           = pVideoName
        self._trkType           = pTrkType
        self._trackedId         = list(self._trkDataLake._data.values())[-1][0][4]


    def computePrimitiveMetrics   (self):
        trk_trajectory              = self._trkDataLake.takeTrajectory(self._trackedId)
        gt_trajectory               = self._gtDataLake.takeTrajectory(self._trackedId)

        first_frame                 = gt_trajectory[0]
        end_frame                   = gt_trajectory[-1]

        self._tp                    = 0
        self._fp                    = 0
        self._fn                    = 0


        for key, gt_values in zip(self._gtDataLake._data.keys(),self._gtDataLake._data.values()):

            if (key >= first_frame and key <= end_frame):

                filtered_gt_values  = list( filter( lambda x: x[4] == self._trackedId ,gt_values ) )
                fn_check            = key not in self._trkDataLake._data.keys()

                if (fn_check == False):
                    trk_values      = self._trkDataLake._data[key]

                #Controllo Falso Negativo
                if (fn_check == True):
                    self._fn += 1
                    continue
                

                gtBox               = list( map( lambda x: x[:4], filtered_gt_values ) )
                gtBox               = gtBox[0]

                
                trkBox              = list( map( lambda x: x[:4], trk_values ) )
                trkBox              = trkBox[0]
                

                iou                 = self._jaccard(gtBox,trkBox)

                #Check TP vs FP
                if (iou >= 0.5):
                    self._tp +=1

                    self._ious.append(iou)
                else:
                    self._fp += 1
 

        self._nFrameTrk             = len(gt_trajectory)

    
    def computeComplexMetrics (self):
        if (self._ious != []):
            cumulator           = float(functools.reduce(lambda a,b : a+b,self._ious))

        self._fpr           = float(self._fp)/self._nFrameTrk
        self._precision     = float(self._tp) / (self._tp + self._fp)

        if ((self._tp + self._fn) != 0):
            self._recall        = float(self._tp) / (self._tp + self._fn)
        else:
            self._recall        = 0.0

        self._sota          = 1 - ( float( ( self._fn + self._fp ) ) / self._nFrameTrk ) 
        self._trkQuality    = self._computeTrackQuality()


        if (self._ious != []):
            self._sotp          = cumulator / self._tp
        else:
            self._sotp          = 0.0


    def exportResults           (self):
        logging.basicConfig(filename=self._exportPath, format='%(asctime)s %(message)s', filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logger.info("\t\t***********SINGLE OBJECT TRACKING EVALUATION (SBTE)**************")
        logger.info("\t\t-> GT_PATH: {}".format(self._gtPath))
        logger.info ("\t\t-> TRK_PATH: {}".format(self._trkPath))
        logger.info ("\t\t-> VIDEO_NAME: {}".format(self._vidName))
        logger.info ("\t\t-> TRK_TYPE: {}".format(self._trkType))
        logger.info ("\t\t-> EXPORT PATH: {}".format(self._exportPath))
        logger.info ("\t\t*****************************************************************")

        logger.info(" ")

        logger.info ("\t\t***********PRIMITIVE METRICS*************************************")
        logger.info ("\t\t-> Number of Tracked Frames: {}".format(self._nFrameTrk))
        logger.info ("\t\t-> True Positive: {}".format(self._tp))
        logger.info ("\t\t-> False Positive: {}".format(self._fp))
        logger.info ("\t\t-> False Negative: {}".format(self._fn))
        logger.info ("\t\t*****************************************************************")

        logger.info(" ")

        logger.info ("\t\t***********COMPLEX METRICS***************************************")
        logger.info ("\t\t-> False Positive Rate: {}".format(self._fpr))
        logger.info ("\t\t-> Precision: {}".format(self._precision))
        logger.info ("\t\t-> Recall: {}".format(self._recall))
        logger.info ("\t\t-> SOTA: {}".format(self._sota))
        logger.info ("\t\t-> SOTP: {}".format(self._sotp))
        logger.info ("\t\t-> Track Quality:\n\t\t\t\t\t\t\t\t\t>>> Tracked Trajectory: {}\n\t\t\t\t\t\t\t\t\t>>> Not Tracked Trajectory: {}\n\t\t\t\t\t\t\t\t\t>>> Fragments: {} ".format(self._trkQuality[0], self._trkQuality[1], self._trkQuality[2]))
        logger.info ("\t\t*****************************************************************")


    def _computeTrackQuality (self):
        target_track        = []
        target_lost         = []

        last_status         = None
        fragment_counter    = 0

        gt_trajectory       = self._gtDataLake.takeTrajectory(self._trackedId)
        first_frame         = gt_trajectory[0]
        end_frame           = gt_trajectory[-1]

        for key, gt_values in zip(self._gtDataLake._data.keys(),self._gtDataLake._data.values()):

            if (key >= first_frame and key <= end_frame):
                fn_check            = key not in self._trkDataLake._data.keys()

                if (fn_check == True):
                    target_lost.append(key)
                    last_status = "Lost"
                else:
                    target_track.append(key)

                    if (last_status == "Lost"):
                        fragment_counter += 1
                        last_status      = None
        
        tracked = ( float(len( target_track ) ) / self._nFrameTrk ) * 100
        lost    = ( float(len( target_lost ) ) / self._nFrameTrk )  * 100

        
        return ( tracked, lost, fragment_counter )


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

        dn = float(boxAArea + boxBArea - interArea)
        if (dn == 0.0):
            return 1.0

        #Calcolo Jaccard
        iou = interArea / dn

        return iou


    def _buildRealExportPath (self):
        base                = self._exportPath
        fl_name, ext        = base.split("-")
        
        timestamp           = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self._exportPath    = fl_name + "-" + timestamp + ext
        

