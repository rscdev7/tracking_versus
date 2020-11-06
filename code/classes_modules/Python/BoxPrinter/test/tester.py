"""
@author: rscalia
@date: Mon 29/06/2020

Questo componente serve per testare la classe BoxPrinter.

"""

import sys
sys.path.append('../lib')
sys.path.append('../extlib')

import os
import sys
import cv2

from BoxPrinter import *


PALETTE_WIDTH  = 3
THICKNESS      = 4


fake_boxes = {}
key_1 = ("SORT",1)
key_2 = ("SORT",2)
key_3 = ("SORT",3)
fake_boxes[ key_1 ] = [200,200,350,350]
fake_boxes[ key_2 ] = [600,500,800,700]
fake_boxes[ key_3 ] = [700,500,1000,700]


img = cv2.imread("../data/img.jpg")
printer = BoxPrinter(PALETTE_WIDTH,THICKNESS)

printer.compute(img,fake_boxes)


fake_boxes[ key_1 ] = [300,200,450,350]

printer.compute(img,fake_boxes)


cv2.imwrite("../data/comp.jpg",img)

