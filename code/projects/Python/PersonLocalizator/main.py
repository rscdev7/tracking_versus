"""
@author    :     rscalia
@date      :     Tue 04/08/2020

Questo componente avvia l'applicativo PersonLocalizer

"""

import wx
import sys
sys.path.append('lib')

from GUI import *

def main ():
    app = wx.App()

    frame = GUI(None, title='PersonLocalizer')
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
