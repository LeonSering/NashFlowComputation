# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         thinFlow_mainControl.py
# Description:  Starts up application if called directly 
# ===========================================================================

if __name__ == '__main__':
    import os
    import sys
    import source.thinFlow_application as application

    cwd = os.getcwd()
    if os.path.basename(cwd) != 'source':
        # If working directory is not 'source', change it
        newCwd = os.path.join(cwd, 'source')
        os.chdir(newCwd)

    app = application.QtWidgets.QApplication(sys.argv)
    form = application.Interface()
    form.show()
    app.exec_()
