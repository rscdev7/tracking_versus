"""
@author: rscalia
@date: Fri 17/04/2020

Questo componente serve per testare la classe DataManager.

"""

import sys
import os
import cv2
from matplotlib import pyplot as plt

sys.path.append('../lib')
sys.path.append('../extlib')

from DataManager import *

#Parametri Programma
DATA_PATH  = "../data/gt.txt"
IMG_PATH   = "../data/img.jpg"
WRITE_PATH = "../data/detections.txt"
THRESHOLD  = 1.0
SYSTEM     = "Inc"
LIMIT      = None
VIEW_LIMIT = 2

detector = DataManager (DATA_PATH, THRESHOLD, LIMIT, SYSTEM, WRITE_PATH)
detector.readData()


#TEST LETTURA DATI
keys = list(detector._data.keys())
keys = keys[:VIEW_LIMIT]


print ("\n###########################TEST LETTURA##################################\n")

for frame_number in keys:
    print ("Frame Number: {} \n".format(frame_number))

    for idx,lis in enumerate(detector._data[frame_number]):
        print ("\nOBJECT N {} \n".format(idx+1))
        print ("-> X Upper Left: {} ".format(lis[0]))
        print ("-> Y Upper Top: {} ".format(lis[1]))
        print ("-> Width: {} ".format(lis[2]))
        print ("-> Height: {} ".format(lis[3]))
        print ("-> Track ID: {} ".format(lis[4]))
        print("\n")

    print("\n")



#TEST STAMPA BOUNDING-BOX
img = cv2.imread(IMG_PATH)
plt.imshow(img[...,::-1])
plt.show()


for rec in detector._data[1]:

    x = rec[0]
    y = rec[1]
    W = rec[2]
    H = rec[3]
    cv2.rectangle(img,(x,y), (W,H),(255,0,0),4)


plt.imshow(img[...,::-1])
plt.show()


#TEST TRAIETTORIA
print ("\n##################################TEST TRAIETTORIA########################\n")
print (detector._data[1])
print("\n")
ris = detector.takeTrajectory(5)
print (ris)


#TEST ORDINE CHIAVI DIZIONARIO
print ("\n###############################TEST ORDINE BBOX NEL DICT######################\n")
print (list(detector._data.keys())[:10])


#TEST SCRITTURA DETECTION
print ("\n############################TEST SCRITTURA#########################\n")
f_number = 20
box = [[20,20,30,40,1], [20,20,30,40,2], [20,20,30,40,3]]
system = "WH"

for b in box:
    idx = b[4]
    detector.write(f_number,idx,b,system)


f_number = 50
box = [[20,20,30,40,1], [20,20,30,40,2], [20,20,30,40,3]]
system = "Inc"

for b in box:
    idx = b[4]
    detector.write(f_number,idx,b,system)


#TEST TARGET DISPONIBILI
print ("\n############################TEST TARGET DISPONIBILI#########################\n")
detector.getAllTargets()
data = detector._avaibleTargets

print (sorted(data.keys()))