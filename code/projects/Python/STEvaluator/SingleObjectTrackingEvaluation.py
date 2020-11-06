"""
@author    :     rscalia
@date      :     Wed 29/07/2020

Questo componente serve per valutare un Single-Object Tracker.

"""

import sys
sys.path.append('./lib')
sys.path.append('./extlib/')
import os

from STEvaluator import *



GT_PATH         = "./data/gt.txt"
TRK_PATH        = "./data/vid_20_TLD_target_5.txt"
VIDEO_NAME      = "03"
TRK_TYPE        = "TLD"
EXPORT_PATH     = "./log/evaluation-.log"



s_eval = STEvaluator(GT_PATH,TRK_PATH,VIDEO_NAME,TRK_TYPE,EXPORT_PATH)
s_eval.computePrimitiveMetrics()
s_eval.computeComplexMetrics()
s_eval.exportResults()


print ("\n***********SINGLE OBJECT TRACKING EVALUATION (SBTE)**************")
print ("-> GT_PATH: {}".format(GT_PATH))
print ("-> TRK_PATH: {}".format(TRK_PATH))
print ("-> VIDEO_NAME: {}".format(VIDEO_NAME))
print ("-> TRK_TYPE: {}".format(TRK_TYPE))
print ("-> EXPORT PATH: {}".format(EXPORT_PATH))
print ("*****************************************************************")

print ("\n***********PRIMITIVE METRICS*************************************")
print ("-> Number of Tracked Frames: {}".format(s_eval._nFrameTrk))
print ("-> True Positive: {}".format(s_eval._tp))
print ("-> False Positive: {}".format(s_eval._fp))
print ("-> False Negative: {}".format(s_eval._fn))
print ("*****************************************************************")

print ("\n***********COMPLEX METRICS***************************************")
print ("-> False Positive Rate: {}".format(s_eval._fpr))
print ("-> Precision: {}".format(s_eval._precision))
print ("-> Recall: {}".format(s_eval._recall))
print ("-> SOTA: {}".format(s_eval._sota))
print ("-> SOTP: {}".format(s_eval._sotp))
print ("-> Track Quality:\n>>>              Tracked Trajectory: {}\n>>>              Not Tracked Trajectory: {}\n>>>              Fragments: {} ".format(s_eval._trkQuality[0], s_eval._trkQuality[1], s_eval._trkQuality[2]))
print ("*****************************************************************")


