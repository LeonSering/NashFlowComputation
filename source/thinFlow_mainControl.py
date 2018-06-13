# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         thinFlow_mainControl.py
# Description:  Starts up application if called directly 
# ===========================================================================

if __name__ == '__main__':
    import sys
    from source import thinFlow_application as application

    app = application.QtGui.QApplication(sys.argv)
    form = application.Interface()
    form.show()
    app.exec_()
