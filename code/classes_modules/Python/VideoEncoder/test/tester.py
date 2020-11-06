"""
@author: rscalia
@date: Fri 26/06/2020

Questo componente serve per testare la classe VideoEncoder.

"""

import os
import sys

sys.path.append('../lib')
sys.path.append('../extlib')

from VideoEncoder import *


#Parametri Programma
BASE        = "../data/2DMOT2015"
TYPE_OF_SET = "test"
CLIP        = "Venice-1"
DATA_FL     = "img1"

IMG_SOURCE = os.path.join(BASE, TYPE_OF_SET, CLIP, DATA_FL)
IMG_FORMAT = ".jpg"
VID_DEST   = os.path.join("../data","2DMOT2015 - VIDEO", TYPE_OF_SET, CLIP+".mp4")

CODEC      = ['F', 'M', 'P', '4']
FPS        = 30
WIDTH      = 1920
HEIGHT     = 1080


enc = VideoEncoder (IMG_SOURCE, IMG_FORMAT,VID_DEST,CODEC,FPS,WIDTH, HEIGHT)
enc.compute()