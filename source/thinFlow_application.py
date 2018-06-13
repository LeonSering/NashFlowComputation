# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         thinFlow_application.py
# Description:  Interface class; controlling signals/slots & communication between widgets
# ===========================================================================

from PyQt4 import QtGui, QtCore
from warnings import filterwarnings
from ui import thinFlow_mainWdw

# =======================================================================================================================
filterwarnings('ignore')  # For the moment: ignore warnings as pyplot.hold is deprecated
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)  # This is necessary if threads access the GUI


class Interface(QtGui.QMainWindow, thinFlow_mainWdw.Ui_MainWindow):
    """Controls GUI"""

    def __init__(self):
        """Initialization of Class and GUI"""
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)

