"""
@author    :     rscalia
@date      :     Wed 29/07/2020

Questo componente serve per implementare una suite di valutazione per un Tracker Multi-Object.

"""

import sys
sys.path.append('../extlib')
import os
import logging
import functools
from scipy.optimize import linear_sum_assignment
import numpy as np 

from DataManager import *
from datetime import *


EMPTY_INTEGER = -1
EMPTY_STRING  = ""
EMPTY_FLOAT   = -1.0
EMPTY_TUPLE   = (EMPTY_FLOAT, EMPTY_FLOAT, EMPTY_INTEGER)


class MTEvaluator (object):

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
        self._nTargetsTrack     = EMPTY_INTEGER
        self._nFrameTrack       = EMPTY_INTEGER
        self._ious              = []
        self._idsw              = EMPTY_INTEGER


        #Metriche Complesse
        self._precision         = EMPTY_FLOAT
        self._recall            = EMPTY_FLOAT
        self._fpr               = EMPTY_FLOAT
        self._mota              = EMPTY_FLOAT
        self._trkQuality        = EMPTY_TUPLE
        self._motp              = EMPTY_FLOAT
        self._weightedIdsw      = EMPTY_FLOAT


        #Path per esportare un log riassuntivo delle prestazione del tracker in esame
        self._exportPath        = pExportPath
        self._buildRealExportPath()


        #Informazioni sul Tracker e sul Video Elaborato
        self._vidName           = pVideoName
        self._trkType           = pTrkType



    def computePrimitiveMetrics   (self):
        self._tp                    = 0
        self._fp                    = 0
        self._fn                    = 0
        self._idsw                  = 0
        self._lastAssign            = {}
        self._nTargetsTrack         = 0
        self._nFrameTrack           = len(list(self._gtDataLake._data.keys()))


        for key, gt_values in zip(self._gtDataLake._data.keys(),self._gtDataLake._data.values()):
            
            self._nTargetsTrack += len(gt_values)


            #Prelievo Dati per effettuare i conteggi
            targets     = gt_values

            if (key in self._trkDataLake._data.keys()):

                hypothesis  = self._trkDataLake._data[key]


                #Eliminazione e Conteggio Falsi Positivi
                to_delete   = self._findFP(targets, hypothesis)
                cleaned_hypothesis = []

                for idx,el in enumerate(hypothesis):
                    if (idx not in to_delete):
                        cleaned_hypothesis.append(el)

                hypothesis = cleaned_hypothesis


                #Conteggio Veri Positivi e Falsi Negativi
                if (len(hypothesis) > 0):

                    #Costruzione Matrice Assegnamento
                    mtr                 = self._buildAssignmentMatrix(targets, hypothesis)

                    #Risoluzione Problema Assegnamento
                    row_ind, col_ind    = linear_sum_assignment(mtr , maximize=True)


                    #Conteggio Falsi Negativi
                    fn_num    = len(targets) - len(col_ind)
                    if (fn_num > 0):
                        self._fn += fn_num
                    

                    #Conteggio Veri Positivi
                    self._tp += len(col_ind)



                    #Check IDSW
                    for row, col in zip(row_ind,col_ind):
                        
                        #Cumulazione Valore Jaccard x metrica MOTP
                        self._ious.append ( mtr[row,col] )


                        #Check per gli IDSW
                        idHypo = hypothesis[row][4]
                        idTarg = targets[col][4]

                        if (idTarg not in self._lastAssign.keys()):
                            self._lastAssign[idTarg] = idHypo
                        else:
                            if (self._lastAssign[idTarg] != idHypo):
                                self._idsw += 1
                                self._lastAssign[idTarg] = idHypo
                else:
                    self._fn += len(targets)
            else:
                self._fn += len(targets)



    def computeComplexMetrics (self):
        if (self._ious != []):
            cumulator           = float(functools.reduce(lambda a,b : a+b,self._ious))

        self._fpr           = float(self._fp)/self._nTargetsTrack
        self._precision     = float(self._tp) / (self._tp + self._fp)
        self._recall        = float(self._tp) / (self._tp + self._fn)

        if (self._recall != 0):
            self._weightedIdsw  = float(self._idsw) / self._recall
        else:
            self._weightedIdsw = self._idsw

        self._mota          = 1 - ( float( ( self._fn + self._fp + self._idsw ) ) / self._nTargetsTrack ) 
        self._trkQuality    = self._computeTrackQuality()

        if (self._ious != []):
            self._motp          = cumulator / self._tp
        else:
            self._motp          = 0.0


    def exportResults           (self):
        logging.basicConfig(filename=self._exportPath, format='%(asctime)s %(message)s', filemode='w')
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        logger.info("\t\t***********MULTI OBJECT TRACKING EVALUATION (MBTE)**************")
        logger.info("\t\t-> GT_PATH: {}".format(self._gtPath))
        logger.info ("\t\t-> TRK_PATH: {}".format(self._trkPath))
        logger.info ("\t\t-> VIDEO_NAME: {}".format(self._vidName))
        logger.info ("\t\t-> TRK_TYPE: {}".format(self._trkType))
        logger.info ("\t\t-> EXPORT PATH: {}".format(self._exportPath))
        logger.info ("\t\t*****************************************************************")

        logger.info(" ")

        logger.info ("\t\t***********PRIMITIVE METRICS*************************************")
        logger.info ("\t\t-> Number of Tracked Frames: {}".format(self._nFrameTrack))
        logger.info ("\t\t-> Number of Tracked Targets: {}".format(self._nTargetsTrack))
        logger.info ("\t\t-> True Positive: {}".format(self._tp))
        logger.info ("\t\t-> False Positive: {}".format(self._fp))
        logger.info ("\t\t-> False Negative: {}".format(self._fn))
        logger.info ("\t\t-> Numero IDSW: {}".format(self._idsw))
        logger.info ("\t\t*****************************************************************")

        logger.info(" ")

        logger.info ("\t\t***********COMPLEX METRICS***************************************")
        logger.info ("\t\t-> False Positive Rate: {}".format(self._fpr))
        logger.info ("\t\t-> Precision: {}".format(self._precision))
        logger.info ("\t\t-> Recall: {}".format(self._recall))
        logger.info ("\t\t-> MOTA: {}".format(self._mota))
        logger.info ("\t\t-> MOTP: {}".format(self._motp))
        logger.info ("\t\t-> Weighted IDSW: {}".format(self._weightedIdsw))
        logger.info ("\t\t-> Track Quality:\n\t\t\t\t\t\t\t\t\t>>> Tracked Trajectory: {}\n\t\t\t\t\t\t\t\t\t>>> Not Tracked Trajectory: {}\n\t\t\t\t\t\t\t\t\t>>> Fragments: {} ".format(self._trkQuality[0], self._trkQuality[1], self._trkQuality[2]))
        logger.info ("\t\t*****************************************************************")


    def _computeTrackQuality (self):
        target_track        = []
        target_lost         = []

        frag_detector       = {}
        fragment_counter    = 0

        
        for key, gt_values in zip(self._gtDataLake._data.keys(),self._gtDataLake._data.values()):
            
            #Prelievo Dati per effettuare i conteggi
            targets     = gt_values

            if (key in self._trkDataLake._data.keys()):
                hypothesis  = self._trkDataLake._data[key]


                #Eliminazione e Conteggio Falsi Positivi
                to_delete   = self._findFP(targets, hypothesis)

                cleaned_hypothesis = []
                
                for idx,el in enumerate(hypothesis):
                    if (idx not in to_delete):
                        cleaned_hypothesis.append(el)

                hypothesis = cleaned_hypothesis


                #Conteggio Veri Positivi e Falsi Negativi
                if (len(hypothesis) > 0):

                    #Costruzione Matrice Assegnamento
                    mtr                 = self._buildAssignmentMatrix(targets, hypothesis)

                    #Risoluzione Problema Assegnamento
                    row_ind, col_ind    = linear_sum_assignment(mtr , maximize=True)



                    #Conteggio Tracker in stato di Lost
                    fn_num    = len(targets) - len(col_ind)
                    if (fn_num > 0):

                        #Conteggio Targets in Stato di Lost
                        for c in range(fn_num):
                            target_lost.append (1)

                        #Aggiornamento Calcolo Frammentazioni Traiettorie
                        set_col_ind = set (col_ind)
                        set_targets = set( list( range( len(targets) ) ) )

                        lost_targets = set_targets.difference(set_col_ind)

                        for idx in lost_targets:
                            ident =  targets[idx][4]
                            frag_detector[ident] = "L"
                    

                    #Conteggio Tracker in stato di Track
                    for i in range(len(col_ind)):
                        target_track.append(1)

                    #Impostazione dei target tracciati a Track 
                    for t in col_ind:
                        idf = targets[t][4]

                        if (idf in frag_detector.keys() and frag_detector[idf] == "L"):
                            fragment_counter += 1

                        frag_detector[idf] = "T"

                else:

                    #Conteggio Tracker in stato di Lost
                    for t in targets:
                        frag_detector[t[4]] = "L"
                    
                    for t in targets:
                        target_lost.append (1)
            else:

                #Conteggio Tracker in stato di Lost
                for t in targets:
                    frag_detector[t[4]] = "L"
                
                for t in targets:
                    target_lost.append (1)
                

        #Calcolo Valori Finali
        tracked      = ( float( len( target_track ) ) / self._nTargetsTrack ) * 100
        lost         = ( float( len( target_lost  ) ) / self._nTargetsTrack )  * 100

        
        return ( tracked, lost, fragment_counter )


    def _findFP (self, pTargets, pHypothesis):
        to_delete = []

        for idx,h in enumerate(pHypothesis):
            inter_o_unions = []

            for t in pTargets:
                inter_o_unions.append( self._jaccard( h[:4], t[:4] ) )
            
            res = list ( filter ( lambda x: x>=0.5, inter_o_unions ) )

            if (res == []):
                to_delete.append(idx)
                self._fp += 1
        
        return to_delete


    def _buildAssignmentMatrix (self, pTargets, pHypothesis):
        mtr = np.zeros ( ( len(pHypothesis) , len(pTargets) ) )

        for rows,h in enumerate(pHypothesis):
            for cols,t in enumerate(pTargets):
                iou = self._jaccard( h[:4], t[:4] ) 
                mtr[rows,cols]  = iou
        
        return mtr



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

