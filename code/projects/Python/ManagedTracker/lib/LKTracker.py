"""
@author: rscalia
@date: Tue 21/07/2020

Questo componente implementa un tracker basato sull'algoritmo di Lucas e Kanade.
"""

import sys
import os
import cv2
import numpy as np
from scipy.spatial import distance

LK_CODE = "Lucas-Kanade"

class LKTracker (object):

    def __init__ (self, pImg ,pBbox, pIDX, pPointsModulator, pLKParams, pFTParamrs ):

        self._lkParams                          = pLKParams
        self._featureParams                     = pFTParamrs

        # FORMAT BBOX --> X, Y, X_MAX, Y_MAX
        self._currentPosition                   = {}
        self._idx                               = pIDX
        self._key                               = (LK_CODE, self._idx)

        self._currentPosition[self._key]        = [ pBbox[0], pBbox[1], pBbox[2], pBbox[3] ]


        self._oldFrameGray                      = cv2.cvtColor(pImg, cv2.COLOR_BGR2GRAY)
        self._pointsModulator                   = pPointsModulator
        

    
    def compute (self, pImg, pBoxes):

        if (type(self._currentPosition) != type(None)):
            self._computeStartPoints()

            if (type(self._oldPoints) != type(None)):

                current_frame_gray      = cv2.cvtColor(pImg, cv2.COLOR_BGR2GRAY)

                new_points, st, err     = cv2.calcOpticalFlowPyrLK(self._oldFrameGray, current_frame_gray, self._oldPoints, None, **self._lkParams)

                good_new = new_points[st==1]
                good_old = self._oldPoints[st==1]


                if (len(good_new) == 0 ):
                    self._currentPosition   = None
                else:
                    x_trl                   = []
                    y_trl                   = []

                    for good_old,good_new in zip(good_old,good_new):
                        x_old, y_old        = good_old.ravel()
                        x_new, y_new        = good_new.ravel()

                        diff_x              = x_new - x_old
                        diff_y              = y_new - y_old

                        x_trl.append( diff_x  )
                        y_trl.append( diff_y )

                    x_trl                   = sorted(x_trl)
                    y_trl                   = sorted (y_trl)

                    mid_x                   = len(x_trl) // 2
                    delta_x                 = (x_trl[mid_x] + x_trl[~mid_x]) / 2

                    mid_y                   = len(y_trl) // 2
                    delta_y                 = (y_trl[mid_y] + y_trl[~mid_y]) / 2


                    proposed_box            = [ self._currentPosition[self._key][0] + delta_x, 
                                                self._currentPosition[self._key][1] + delta_y, 
                                                self._currentPosition[self._key][2] + delta_x, 
                                                self._currentPosition[self._key][3] + delta_y]

                    vote                    = []
                    for idx,box in enumerate(pBoxes):
                        overlap             = self._jaccard(proposed_box, box)

                        #CALCOLO CENTROIDI
                        p1_x                =  box[0] + box[2]//2
                        p1_y                =  box[1] + box[3]//2
                        p1                  = [p1_x,p1_y]

                        p2_x                =  proposed_box[0] + proposed_box[2]//2
                        p2_y                =  proposed_box[1] + proposed_box[3]//2
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
                        self._currentPosition = None
                    else: 
                        restricted_vote     = list( map (lambda x: (x[0],x[2]) , restricted_vote) )
                        final_vote          = sorted(restricted_vote, key=lambda x: x[1])
                        verdict             = final_vote[0][0]

                        self._currentPosition[self._key][0] = pBoxes[verdict][0]
                        self._currentPosition[self._key][1] = pBoxes[verdict][1]
                        self._currentPosition[self._key][2] = pBoxes[verdict][2]
                        self._currentPosition[self._key][3] = pBoxes[verdict][3]


                        self._oldFrameGray                  = current_frame_gray.copy()
            else:
                self._currentPosition                       = None

    def _computeStartPoints (self):

        #Generazione Punti di Partenza Optical Flow
        if (self._pointsModulator == None ):
            m                       = np.zeros_like(self._oldFrameGray)
            x                       = int(self._currentPosition[self._key][0])
            y                       = int(self._currentPosition[self._key][1])
            x_max                   = int(self._currentPosition[self._key][2])
            y_max                   = int(self._currentPosition[self._key][3])

            m[y:y_max, x:x_max]     = 255

            
            self._oldPoints               = cv2.goodFeaturesToTrack(self._oldFrameGray, mask = m, **self._featureParams)
        else:
            self._oldPoints               = self._spacedSample()


    def _spacedSample (self):

        s = self._currentPosition[self._key][0]
        e = self._currentPosition[self._key][2]
        step = (e -s) //self._pointsModulator

        s2 = self._currentPosition[self._key][1]
        e2 = self._currentPosition[self._key][3]
        step2 = (e2 -s2) //self._pointsModulator


        xlspace  = np.array(np.linspace (s,e,step, dtype = int))
        ylspace  = np.array(np.linspace (s2,e2,step2, dtype = int))

        arr      = None
        if (step >= step2):
            arr = self._generatePoints(xlspace,ylspace)
        else:
            arr = self._generatePoints(ylspace,xlspace)

        return arr

    def _generatePoints (self, major_axis, minor_axis):
        major_len   = len(major_axis)
        minor_len   = len(minor_axis)
        arr         = list()

        for i in range (major_len):
            for j in range (minor_len):
                arr.append( [ [major_axis[i] , minor_axis[j] ] ] )

        arr = np.array(arr, np.dtype('float32'))

        return arr

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