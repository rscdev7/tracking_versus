"""
@author    :     rscalia
@date      :     Tue 04/08/2020

Questo componente permette di impiegare un'interfaccia grafica all'interno dell'applicativo PersonLocalizer.

"""

import wx
import cv2
import sys
sys.path.append('./extlib')

from DataManager import *
from ManagedCentroid import *
from ManagedCSRT import *
from ManagedSORT import *
from LKTracker   import *
from BoxPrinter import *
from TrackingLogic import *
from imutils.video import FPS

class GUI (wx.Frame):

    def __init__(self, parent, title="PersonLocalizer"):
        super(GUI, self).__init__(parent, title=title)

        self._algorithms    = ["SORT", "CSRT", "Centroid Tracker", "Lucas-Kanade Tracker"]
        self._selectedAlgo  = None

        self.capture        = None
        self._fCounter      = 0
        self._videos        = {}
        self._dataPath      = None
        self._fps           = None
        self._frameSize     = None
        self._fillVideos()
        self._selectedVid   = None

        self._logic         = TrackingLogic()
        self._Tracking      = False


        self._fpsec           = None
        self.setUI()


    def setUI(self):
        self.SetTitle('PersonLocalizer')
        self.SetSize( (1280,720) )
        
        
        #Elementi UI
        self._pnl 		         = wx.Panel(self)

        self._mainSizer 		 = wx.BoxSizer(wx.VERTICAL)
        self._commandsSizer      = wx.BoxSizer(wx.HORIZONTAL)

        self._playerSizer        = wx.BoxSizer(wx.VERTICAL)
        self._trackerSizer       = wx.BoxSizer(wx.VERTICAL)



        #Filling Video Player UI
        self.image 		= wx.EmptyImage(1280,720)
        self.imageBit 	= wx.BitmapFromImage(self.image)
        self.staticBit 	= wx.StaticBitmap(self._pnl, wx.ID_ANY, self.imageBit)

        
        #Filling Video Player Buttons UI Elements
        self._play          = wx.Button(self._pnl, label="PLAY")
        self._stop          = wx.Button(self._pnl, label="STOP")
        self._reset         = wx.Button(self._pnl, label="RESET")
        self._selectorVid   = wx.ComboBox(self._pnl, choices=list(self._videos.keys()), 
            style=wx.CB_READONLY)

        #Filling Video Player Buttons Sizer
        self._playerSizer.Add(self._play ,0,wx.CENTER|wx.ALL,10)
        self._playerSizer.Add(self._stop,0,wx.CENTER|wx.ALL,10)
        self._playerSizer.Add(self._reset,0,wx.CENTER|wx.ALL,10)
        self._playerSizer.Add(self._selectorVid,0,wx.CENTER|wx.ALL,10)


        #Filling Tracker UI Elements
        self._roi      = wx.Button(self._pnl, label="ROI / COMPUTE")
        self._selector = wx.ComboBox(self._pnl, choices=self._algorithms, 
            style=wx.CB_READONLY)

        #Filling Tracker Sizer
        self._trackerSizer.Add(self._selector,0,wx.CENTER|wx.ALL,10)
        self._trackerSizer.Add(self._roi,0,wx.CENTER|wx.ALL,10)


        #Composizione Sizer 
        self._commandsSizer.Add(self._playerSizer, wx.ALIGN_RIGHT|wx.ALL)
        self._commandsSizer.Add(self._trackerSizer, wx.ALIGN_LEFT|wx.ALL)

        self._mainSizer.Add(self.staticBit)
        self._mainSizer.Add(self._commandsSizer)


        #Impostazione Pannello Principale
        self._pnl.SetSizer(self._mainSizer)
        self._mainSizer.Fit(self)
        self.Show()


        #Timer Aggiornamento Frame Video
        self.timex          = wx.Timer(self)


        #Bind
        self.Bind(wx.EVT_BUTTON, self.OnPlay, self._play)
        self.Bind(wx.EVT_BUTTON, self.OnStop, self._stop)
        self.Bind(wx.EVT_BUTTON, self.OnReset, self._reset)
        self.Bind(wx.EVT_BUTTON, self.OnRoi, self._roi)
        self.Bind(wx.EVT_COMBOBOX, self.OnSelect, self._selector)
        self.Bind(wx.EVT_COMBOBOX, self.OnVidSel, self._selectorVid)
        self.Bind(wx.EVT_TIMER, self.redraw, self.timex)
 

    def startDiplayVideo (self):
        self.timex.Stop()
        self._logic.init(self._dataPath)
        self._Tracking      = False
        
        self._fCounter      = 0
        self.capture        = cv2.VideoCapture(self._selectedVid)
        self._fps           = int(self.capture.get(cv2.CAP_PROP_FPS))
        self._frameSize     = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))


        ret, self.frame     = self._takeFrame()

        if ret:
            self.frame      = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            if (self._frameSize[0] >= 1920 or self._frameSize[1] >= 1080):
                self.frame      = cv2.resize(self.frame,(1280,720))

                self.bmp        = wx.BitmapFromBuffer(1280, 720, self.frame)
                self.staticBit.SetBitmap(self.bmp)
            else:
                self.bmp        = wx.BitmapFromBuffer(self._frameSize[0], self._frameSize[1], self.frame)
                self.staticBit.SetBitmap(self.bmp)
            
            self.Refresh()

            
    def redraw(self,e):
        ret, self.frame = self._takeFrame()


        if ret:
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            
            if (self._Tracking      == True):
                self.timex.Stop()
                self._logic.compute(self.frame, self._fCounter)
                self.timex.Start(1000./self._fps)


            if (self._frameSize[0] >= 1920 or self._frameSize[1] >= 1080):
                self.frame      = cv2.resize(self.frame,(1280,720))


            self._fpsec.update()
            self._fpsec.stop()

            text = "FPS: {}".format(self._fpsec.fps())

            cv2.putText(self.frame, text, (50,50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


            self.bmp.CopyFromBuffer(self.frame)
            self.staticBit.SetBitmap(self.bmp)
            
        
            self.Refresh()


    def OnPlay (self, e):
        if (self._selectedVid != None):
            if (self.capture == None):
                self.startDiplayVideo()
                
            if (self._fpsec == None):
                self._fpsec = FPS().start()

            self.timex.Start(1000./self._fps)
            
    
    def OnStop (self, e):
        if (self._selectedVid != None and self.capture != None):
            self.timex.Stop()
            self._fpsec.stop()

    def OnReset (self, e):
        if (self._selectedVid != None):
            if (self.capture != None):
                self.capture.release()

            self._fpsec = FPS().start()
            self.startDiplayVideo()


    def OnRoi (self, e):
        if (self._selectedVid != None and self.capture != None and self._selectedAlgo != None):
            self.timex.Stop()
            self._fpsec.stop()

            ret, self.frame = self._takeFrame()

            
            if (ret):
                ROI  = None
                if (self._selectedAlgo != "SORT" !=  self._selectedAlgo != "Centroid Tracker"):
                    initBB = cv2.selectROI("Select ROI", self.frame,fromCenter=False,
                    showCrosshair=True)
                    ROI = (initBB[0], initBB[1], initBB[0] +initBB[2], initBB[1] +initBB[3] )
                key = cv2.waitKey(1)& 0xFF 


                self._Tracking      = True
                self._logic.registerTracker(self._selectedAlgo, ROI, self._fCounter, self.frame)
                
                cv2.destroyAllWindows()

                self.timex.Start(1000./self._fps)
                self._fpsec = FPS().start()


    def OnSelect (self, e):
        self._selectedAlgo = e.GetString()


    def OnVidSel (self, e):
        self._selectedVid               = self._videos[str(e.GetString())][0]
        self._dataPath                  = self._videos[str(e.GetString())][1]

        if (self.capture != None):
            self.capture.release()
            self.capture = None

        self.startDiplayVideo()


    def _takeFrame (self):
        ret, self.frame     = self.capture.read()
        self._fCounter     += 1

        return ret, self.frame


    def _fillVideos (self):
        self._videos["MOT16-02"] = ("./data/video_lake/1/MOT16-02.mp4" , "./data/video_lake/1/gt.txt")
        self._videos["MOT16-04"] = ("./data/video_lake/2/MOT16-04.mp4" , "./data/video_lake/2/gt.txt")
        self._videos["MOT16-05"] = ("./data/video_lake/3/MOT16-05.mp4" , "./data/video_lake/3/gt.txt")
        self._videos["MOT16-09"] = ("./data/video_lake/4/MOT16-09.mp4" , "./data/video_lake/4/gt.txt")
        self._videos["MOT16-10"] = ("./data/video_lake/5/MOT16-10.mp4" , "./data/video_lake/5/gt.txt")
        self._videos["MOT16-11"] = ("./data/video_lake/6/MOT16-11.mp4" , "./data/video_lake/6/gt.txt")
        self._videos["MOT16-13"] = ("./data/video_lake/7/MOT16-13.mp4" , "./data/video_lake/7/gt.txt")
        self._videos["ADL-Rundle-6"] = ("./data/video_lake/8/ADL-Rundle-6.mp4" , "./data/video_lake/8/gt.txt")
        self._videos["ADL-Rundle-8"] = ("./data/video_lake/9/ADL-Rundle-8.mp4" , "./data/video_lake/9/gt.txt")
        self._videos["ETH-Bahnhof"] = ("./data/video_lake/10/ETH-Bahnhof.mp4" , "./data/video_lake/10/gt.txt")
        self._videos["ETH-Pedcross2"] = ("./data/video_lake/11/ETH-Pedcross2.mp4" , "./data/video_lake/11/gt.txt")
        self._videos["ETH-Sunnyday"] = ("./data/video_lake/12/ETH-Sunnyday.mp4" , "./data/video_lake/12/gt.txt")
        self._videos["KITTI-13"] = ("./data/video_lake/13/KITTI-13.mp4" , "./data/video_lake/13/gt.txt")
        self._videos["KITTI-17"] = ("./data/video_lake/14/KITTI-17.mp4" , "./data/video_lake/14/gt.txt")

        self._videos["PETS09-S2L1"] = ("./data/video_lake/15/PETS09-S2L1.mp4" , "./data/video_lake/15/gt.txt")
        self._videos["TUD-Campus"] = ("./data/video_lake/16/TUD-Campus.mp4" , "./data/video_lake/16/gt.txt")
        self._videos["TUD-Stadtmitte"] = ("./data/video_lake/17/TUD-Stadtmitte.mp4" , "./data/video_lake/17/gt.txt")
        self._videos["Venice-2"] = ("./data/video_lake/18/Venice-2.mp4" , "./data/video_lake/18/gt.txt")
        
        