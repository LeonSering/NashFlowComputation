# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         mainControl.py
# Description:  Starts up application if called directly 
# ===========================================================================

if __name__ == '__main__':
    import os
    import sys
    from math import ceil
    from source import application

    cwd = os.getcwd()
    if os.path.basename(cwd) == 'source':
        # If working directory is 'source', change it
        newCwd = os.path.dirname(cwd)
        os.chdir(newCwd)


    def onResize(event, widgetList, referenceSize):
        '''
        Resizes widgets when the main window is being resized
        :param event: resizeEvent passed by signal
        :param widgetList: List of all widgets that need to be resized
        :param referenceSize: original size of main window
        '''
        if (event.oldSize().height(), event.oldSize().width()) == (-1, -1):
            # Initialization should be ignored
            return

        newMainWindowSize = event.size() # New size of main window
        widthToHeightRatio = float(referenceSize.width()) / referenceSize.height()
        fixedHeight = int(ceil(1. / widthToHeightRatio * form.width()))
        sizeDiff = newMainWindowSize - event.oldSize()
        widthDiff, heightDiff = sizeDiff.width(), sizeDiff.height()

        if widthDiff != 0 and heightDiff == 0:
            #Window changed width -> adjust height
            form.setMinimumHeight(fixedHeight)
            form.setMaximumHeight(fixedHeight)
        else:
            # Compute rescaling factor in x and y direction
            xNew, yNew = float(newMainWindowSize.width()), float(newMainWindowSize.height())
            xRescale, yRescale = xNew / referenceSize.width(), yNew / referenceSize.height()

            for widget, size, position, fontPointSize in widgetList:
                # Update the widgets size
                newSize = application.QtCore.QSize(int(ceil(xRescale * size.width())), int(ceil(yRescale * size.height())))
                widget.resize(newSize)

                # Update the widgets position
                newPos = application.QtCore.QPoint(int(ceil(xRescale * position.x())), int(ceil(yRescale * position.y())))
                widget.move(newPos)

                # Update the widgets font size
                newFont = widget.font() # This should be returned as constant?
                newFontSize = int(ceil((0.5 * (xRescale + yRescale) * fontPointSize)))
                newFont.setPointSize(newFontSize)
                widget.setFont(newFont)



    app = application.QtGui.QApplication(sys.argv)
    form = application.Interface()

    # Generate lists of widgets that need to be resized when resizing the main window
    directWidgets = [form.plotFrame, form.plotAnimationFrame, form.plotDiagramFrame, form.plotNTFFrame] # Directly specified
    widgetTypes = [application.QtGui.QGroupBox, application.QtGui.QPushButton, application.QtGui.QLineEdit,
                   application.QtGui.QPlainTextEdit, application.QtGui.QLabel, application.QtGui.QListWidget,
                   application.QtGui.QComboBox, application.QtGui.QCheckBox, application.QtGui.QSlider]

    widgetListbyType = [(widget, widget.size(), widget.pos(), widget.font().pointSize()) for widgetType in widgetTypes for widget in form.findChildren(widgetType)]
    additionalWidgetList = [(widget, widget.size(), widget.pos(), widget.font().pointSize()) for widget in directWidgets]

    widgetList = widgetListbyType + additionalWidgetList

    referenceSize = form.size() # Get reference size of main window

    # Set initial height to avoid automatic shrinking at the beginning
    form.setMinimumHeight(referenceSize.height())
    form.setMaximumHeight(referenceSize.height())

    # Called to update window
    form.resizeEvent = lambda event: onResize(event, widgetList, referenceSize)

    form.show()
    app.exec_()
