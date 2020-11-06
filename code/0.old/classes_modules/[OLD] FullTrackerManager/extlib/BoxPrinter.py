"""
@author: rscalia
@date: Mon 29/06/2020

Questo componente serve per stampare sui frame le boudning-box.

"""

import os
import sys
import seaborn as sns
import cv2


FIRST_COLOR = 0

class BoxPrinter (object):

    def __init__ (self, pPaletteWidth, pThicknessLine, pSystem):

        self._paletteWidth  = pPaletteWidth
        self._colorPalette  = sns.color_palette( "hls" , n_colors = self._paletteWidth)
        self._colorMap      = {}
        self._nextColor     = FIRST_COLOR
        self._ticknessLine  = pThicknessLine
        self._system        = pSystem

        #Denormalizzazione delle terne RGB restituite dalla libreria
        for idx,value in enumerate(self._colorPalette):
            value                   = list( map ( lambda x: int(x*255) , value) )
            self._colorPalette[idx] = value


    #pBoxes FORMAT ---> [ ( ALGO_CODE , ID ) ] = (x_1, y_1, x_2, y_2)
    def compute (self, pImage, pBoxes):
        
        
        #Selezione Colore Bounding-Box
        for key, value in zip(pBoxes.keys() , pBoxes.values()):
            
            if (key in self._colorMap.keys()):

                #Aggiorno BBOX
                self._colorMap[key][0] = value
            else:

                #Aggiungo Nuovo Elemento Tracciato
                self._colorMap[key] = [value, self._colorPalette[self._nextColor]]
                self._nextColor     += 1

                #La palette colori ha un funzionamento circolare
                if (self._nextColor >= self._paletteWidth):
                    self._nextColor = 0


        #Eliminazione Vecchie Bounding-Box
        keys = set (pBoxes.keys())
        color_keys = set(self._colorMap.keys())
        diff = list( color_keys.difference(keys) )

        if (diff != []):
            for key in diff:
                del self._colorMap[key]

            
        #Print delle Bounding-Box
        for value in self._colorMap.values():
            x_1 = int (value[0][0])
            y_1 = int (value[0][1])
            x_2 = int (value[0][2])
            y_2 = int (value[0][3])

            r = value[1][0]
            g = value[1][1]
            b = value[1][2]

            if (self._system == "Inc"):
                cv2.rectangle(pImage, (x_1 , y_1), (x_2 , y_2), (b,g,r), self._ticknessLine)
            else:
                cv2.rectangle(pImage, (x_1 , y_1), (x_1 + x_2 , y_1 + y_2), (b,g,r), self._ticknessLine)
            
            
        
        