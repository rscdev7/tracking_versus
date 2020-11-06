"""
@author    :     rscalia
@date      :     Wed 29/07/2020

Questo componente serve per valutare un Multi-Object Tracker.

"""

import sys
sys.path.append('./lib')
sys.path.append('./extlib/')
import os

from MTEvaluator import *



#Config
GT_PATH         = "./data/gt.txt"
TRK_PATH        = "./data/vid_15_centroid.txt"
VIDEO_NAME      = "PETS09-S2L1"
TRK_TYPE        = "Centroid"
EXPORT_PATH     = "./log/evaluation-.log"



s_eval = MTEvaluator(GT_PATH,TRK_PATH,VIDEO_NAME,TRK_TYPE,EXPORT_PATH)
s_eval.computePrimitiveMetrics()
s_eval.computeComplexMetrics()
s_eval.exportResults()


print ("\n***********MULTI OBJECT TRACKING EVALUATION (MBTE)**************")
print ("-> GT_PATH: {}".format(GT_PATH))
print ("-> TRK_PATH: {}".format(TRK_PATH))
print ("-> VIDEO_NAME: {}".format(VIDEO_NAME))
print ("-> TRK_TYPE: {}".format(TRK_TYPE))
print ("-> EXPORT PATH: {}".format(s_eval._exportPath))
print ("*****************************************************************")

print ("\n***********PRIMITIVE METRICS*************************************")
print ("-> Number of Tracked Frames: {}".format(s_eval._nFrameTrack))
print ("-> Number of Tracked Targets: {}".format(s_eval._nTargetsTrack))
print ("-> True Positive: {}".format(s_eval._tp))
print ("-> False Positive: {}".format(s_eval._fp))
print ("-> False Negative: {}".format(s_eval._fn))
print ("-> Numero IDSW: {}".format(s_eval._idsw))
print ("*****************************************************************")

print ("\n***********COMPLEX METRICS***************************************")
print ("-> False Positive Rate: {}".format(s_eval._fpr))
print ("-> Precision: {}".format(s_eval._precision))
print ("-> Recall: {}".format(s_eval._recall))
print ("-> MOTA: {}".format(s_eval._mota))
print ("-> MOTP: {}".format(s_eval._motp))
print ("-> Weighted IDSW: {}".format(s_eval._weightedIdsw))
print ("-> Track Quality:\n>>>              Tracked Trajectory: {}\n>>>              Not Tracked Trajectory: {}\n>>>              Fragments: {} ".format(s_eval._trkQuality[0], s_eval._trkQuality[1], s_eval._trkQuality[2]))
print ("*****************************************************************")


