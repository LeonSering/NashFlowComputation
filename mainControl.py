# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         mainControl.py
# Description:  Starts up application if called directly 
# ===========================================================================

if __name__ == '__main__':
    import os
    import sys
    import warnings
    from source import application

    warnings.filterwarnings('ignore')

    cwd = os.getcwd()
    if os.path.basename(cwd) == 'source':
        # If working directory is 'source', change it
        newCwd = os.path.dirname(cwd)
        os.chdir(newCwd)

    app = application.QtWidgets.QApplication(sys.argv)
    form = application.Interface()
    form.show()
    app.exec_()
