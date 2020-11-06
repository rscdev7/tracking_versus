import sys
sys.path.append ('../lib')
sys.path.append ('../extlib')

import os
from ManagedTracker import *
import cv2
import numpy as np

TYPE           = 'SORT'
DETECTIONS     = np.array     ( [ [50,60,80,90] , [200,250,400,300] , [150,41,300,120] ]  )
NEW_DETECTIONS = np.array     ( [ [70,60,90,90] , [250,250,450,300] , [200,41,350,120] ]  )
#DETECTIONS     = (50,60,80,90)
#NEW_DETECTIONS = (70,60,90,90)
IMG            = cv2.imread   ("../data/000748.jpg")
IMG_2          = cv2.imread   ("../data/000749.jpg")
INFORMATION    = {
    'detections'    : DETECTIONS, \
    'img'           : IMG, \
    'ID'            : 1
                 }


print ("\n*******************INFO*************************")
print ("-> Track Engine: {} ".format(TYPE))
print ("-> Img Shape: {} ".format(IMG.shape))
print ("\n-> Detection 1:\n {} \n".format(DETECTIONS))
print ("-> Detection 2:\n {} \n".format(NEW_DETECTIONS))
print ("************************************************\n")


trk = ManagedTracker (TYPE , INFORMATION)
print ("\n-> START: \n   -> Data:\n {} \n\n   -> Type: {} \n".format(trk._currentData, type(trk._currentData)))


INFORMATION['detections'] = NEW_DETECTIONS
res  = trk.compute(INFORMATION)

print ("\n-> COMPUTATION: \n   -> Data: \n {} \n\n   -> Type: {} \n".format(res, type(res)))
