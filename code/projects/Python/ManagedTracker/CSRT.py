"""
@author: rscalia
@date: Wed 15/07/2020

Questo serve per testare la classe ManagedCSRT.

"""

import sys
sys.path.append ('./lib')
sys.path.append ('./extlib')

import numpy
import cv2
from DataManager              import *
from ManagedCSRT              import *
from BoxPrinter               import *


GT_PATH             = "./data/gt.txt"
GT_THRESHOLD        = 1.0
GT_LIMIT            = None
GT_SYSTEM           = "WH"
WRITE_DATA_PATH     = "./data/detections.txt"



PALETTE_WIDTH       = 20
THICKNESS_LINE      = 4

VIDEO_IN            = "./data/vid.mp4"
VIDEO_OUT           = "./data/vid_out.mp4"


#Estrazione Bounding-Box per ogni frame
detector = DataManager(GT_PATH , GT_THRESHOLD , GT_LIMIT, GT_SYSTEM, WRITE_DATA_PATH)
detector.readData()
detector.getAllTargets()


print ("Avaible Targets: \n {} \n".format(sorted(detector._avaibleTargets.keys())))
choose = int(input("Scegli il Target \n"))
frame_choice = detector._avaibleTargets[(choose)]

print ("Target Scelto: {} - Frame Iniziale: {} \n".format(choose,frame_choice))
if (choose > max(detector._avaibleTargets.keys()) or choose < min(detector._avaibleTargets.keys())):
    print ("\nBad target choice \n")
    exit(-4)


IDX                 = choose
KEY                 = (CSRT_TOKEN, IDX)

#Configurazione Target
trk_engine = None

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
print_trigger = False


#COMPUTAZIONE....
while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:

        #Log
        t+=1
        print ("Frame Elab: {} - Frame Totali: {} - Secondi Elab: {} - Secondi Rimanenti: {} - Minuti Elab: {} - Minuti Rimanenti: {} \n".format(t,length, t/fps, (length-t)//fps, (t/fps)/60, ((length-t)/fps)//60))


        if (t in detector._data.keys()):
    
            #TrackEngine Inizialization and Computation
            if (t == frame_choice):

                #Prelievo target e filtraggio
                boxes = detector._data[t]
                box = list( filter (lambda x: x[4] == choose, boxes) )
                if (box != []):
                    box = box[0][:4]

                #Init
                trk_engine = ManagedCSRT(frame, box, IDX)
            elif (trk_engine != None):
                #Compute
                trk_engine.computeAndStore(frame)

            #Print BBOX e Logging
            if (type(trk_engine) != type(None) and type(trk_engine._trackInfo[KEY]) != type(None)):

                #Prelievo BBOX
                cp_boxes = trk_engine._trackInfo
                values = cp_boxes[KEY]

                #Logging
                values = [ int(values[0]) , int(values[1]), int(values[2]), int(values[3]) ]
                detector.write(t, IDX, values, GT_SYSTEM)


                #Stampa BBOX
                x_1 = cp_boxes[KEY][0]
                y_1 = cp_boxes[KEY][1]
                x_2 = cp_boxes[KEY][0] + cp_boxes[KEY][2]
                y_2 = cp_boxes[KEY][1] + cp_boxes[KEY][3]

                cp_boxes[KEY] = [x_1, y_1, x_2, y_2]
                printer.compute(frame, cp_boxes)

        ris = out.write(frame)
    else:
        break


#Rilascio le risorse allocate
cap.release()
out.release()
