"""
@author: rscalia
@date: Wed 15/07/2020

Questo serve per testare la classe ManagedSORT.

"""

import sys
sys.path.append ('./lib')
sys.path.append ('./extlib')

import numpy
import cv2
from DataManager          import *
from ManagedSORT          import *
from BoxPrinter           import *



GT_PATH             = "./data/gt.txt"
GT_THRESHOLD        = 1.0
GT_LIMIT            = None
GT_SYSTEM           = "Inc"
WRITE_DATA_PATH     = "./data/detections.txt"

PALETTE_WIDTH       = 20
THICKNESS_LINE      = 4

VIDEO_IN            = "./data/vid.mp4"
VIDEO_OUT           = "./data/vid_out.mp4"



#Estrazione Bounding-Box per ogni frame
detector = DataManager(GT_PATH , GT_THRESHOLD , GT_LIMIT, GT_SYSTEM, WRITE_DATA_PATH)
detector.readData()


#TrackEngine Inizialization
trk_engine = ManagedSORT ()


#Inizializzazione Printer
printer    = BoxPrinter(PALETTE_WIDTH , THICKNESS_LINE)



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

            trk_engine.computeAndStore(boxes)

            cp_boxes = trk_engine._trackInfo

            #LOGGING
            for key, values in zip(cp_boxes.keys(), cp_boxes.values()):
                idx = key[1]
                values = [ int(values[0]) , int(values[1]), int(values[2]), int(values[3]) ]
                
                detector.write(t, idx, values, GT_SYSTEM)

            printer.compute(frame, cp_boxes)

        ris = out.write(frame)

    else:
        break


#Rilascio le risorse allocate
cap.release()
out.release()
