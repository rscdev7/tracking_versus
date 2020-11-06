"""
@author: rscalia
@date: Fri 26/06/2020

Questo componente serve codificare in un video una serie di immagini.

"""

import os
import sys
import cv2

READ_ERROR = -1

class VideoEncoder (object):


    def __init__ (self, pImgSource, pImgFormat,pVidDest, pCodec, pFps, pWidth, pHeight):

        self._imgSource = pImgSource
        self._imgFormat = pImgFormat
        self._vidDest   = pVidDest
        self._codec     = pCodec
        self._fps       = pFps
        self._width     = pWidth
        self._height    = pHeight


    #Codifica Video
    def compute (self):

        #VIDEO OUT
        out = cv2.VideoWriter(self._vidDest, cv2.VideoWriter_fourcc(self._codec[0], self._codec[1], self._codec[2], self._codec[3]), self._fps, (self._width, self._height))

        if (out.isOpened()== False): 
            print("Errore nell'apertura file di scrittura")
            return READ_ERROR

        img_list = []

        for file in os.listdir(self._imgSource):
            if file.endswith(self._imgFormat):
                img_list.append(file)

        img_list.sort()
        
        length = len(img_list)

        for img_path in img_list:
            full_path = os.path.join(self._imgSource,img_path)
            im = cv2.imread(full_path)

            ris = out.write(im)
            del im

            c_idx        = img_list.index(img_path)+1
            sec_cp       = c_idx/self._fps
            sec_residual = length / self._fps

            print ("Frame Elaborato: {} - Frame Rimanenti: {} - Secondi Elaborati: {} - Secondi Totali: {} \n".format (c_idx, length, sec_cp, sec_residual))

        out.release()



        
