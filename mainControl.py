# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         mainControl.py
# Description:  Starts up application if called directly 
# ===========================================================================
if __name__ == '__main__':
    import sys
    from source import application

    app = application.QtGui.QApplication(sys.argv)
    form = application.Interface()
    form.show()
    app.exec_()
