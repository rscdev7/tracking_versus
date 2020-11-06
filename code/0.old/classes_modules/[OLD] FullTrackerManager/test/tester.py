"""
@author: rscalia
@date: Fri 03/07/2020

Questo componente serve per .........

"""

import sys
sys.path.append ('../lib')
sys.path.append ('../extlib')

import numpy
import cv2
from GTTracker          import *
from FullTrackerManager import *
from BoxPrinter         import *



GT_PATH             = "../data/gt.txt"
GT_THRESHOLD        = 1.0
GT_LIMIT            = 2
GT_SYSTEM           = "WH"

TRACK_ALGORITHM     = "CSTR"
JACCARD_THR         = 1.0

PALETTE_WIDTH       = 2
THICKNESS_LINE      = 4

VIDEO_IN            = "../data/vid.mp4"
VIDEO_OUT           = "../data/vid_out.mp4"



#Estrazione Bounding-Box per ogni frame
detector = GTTracker(GT_PATH , GT_THRESHOLD , GT_LIMIT, GT_SYSTEM)
detector.compute()


#TrackEngine Inizialization
trk_engine = FullTrackerManager (TRACK_ALGORITHM , JACCARD_THR)


#Inizializzazione Printer
printer    = BoxPrinter(PALETTE_WIDTH , THICKNESS_LINE, GT_SYSTEM)



#Creazione strutture dati per leggere/scrivere sui filmati
cap = cv2.VideoCapture(VIDEO_IN)
if (cap.isOpened()== False): 
    print("Errore nell'apertura del file")
    exit(-1)

length          = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width     = int(cap.get(3))
frame_height    = int(cap.get(4))
fps             = int(cap.get(cv2.CAP_PROP_FPS))


out = cv2.VideoWriter(VIDEO_OUT,cv2.VideoWriter_fourcc('F','M','P','4'), fps, (frame_width,frame_height))
if (out.isOpened()== False): 
    print("Errore nell'apertura file di scrittura")
    exit(-1)


#Contatori
counter = 0
t = 0


#COMPUTAZIONE....
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:

        #Log
        t+=1
        print ("Frame Elab: {} - Frame Totali: {} - Secondi Elab: {} - Secondi Rimanenti: {} - Minuti Elab: {} - Minuti Rimanenti: {} \n".format(t,length, t/fps, (length-t)//fps, (t/fps)/60, ((length-t)/fps)//60))

        if (t in detector._data.keys()):
            boxes = detector._data[t]
            boxes = list ( map ( lambda x: x[:4], boxes  ) )
            trk_engine.compute(frame, boxes)

            cp_boxes = trk_engine._trackInfo
            printer.compute(frame, cp_boxes)

        ris = out.write(frame)

    else:
        break


#Rilascio le risorse allocate
cap.release()
out.release()
