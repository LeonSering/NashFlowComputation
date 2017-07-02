# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotValuesCanvasClass.py
# ===========================================================================

import matplotlib
from matplotlib import figure

matplotlib.use("Qt4Agg")
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas


# ======================================================================================================================



class PlotValuesCanvas(FigureCanvas):
    def __init__(self, callback=None):
        self.figure = figure.Figure()
        super(PlotValuesCanvas, self).__init__(self.figure)  # Call parents constructor

        self.additionalColors = ['red', 'blue']
        self.verticalLine = None
        self.verticalLinePos = 0
        self.callback = callback

        self.visibleBool = False

        # Signals
        self.mpl_connect('button_press_event', self.on_click)



    def update_plot(self, lowerBound, upperBound, labels, xValues, yValues, *additional_values):
        self.figure.clf()

        axes = self.figure.add_subplot(111)

        yMin, yMax = min(yValues), max(yValues)
        axes.plot(xValues, yValues, linewidth=2, color='green', label=labels[0])
        colorCounter = 0
        for xVals, yVals in additional_values:
            yMin, yMax = min(yMin, min(yVals)), max(yMax, max(yVals))
            axes.plot(xVals, yVals, linewidth=2, color=self.additionalColors[colorCounter], label=labels[colorCounter + 1])
            colorCounter += 1

        axes.set_xlim(lowerBound, upperBound)
        axes.set_ylim(max(0, yMin), int(max(1, yMax) * 1.5))

        if lowerBound <= self.verticalLinePos <= upperBound:
            self.verticalLine = axes.axvline(self.verticalLinePos)
            self.verticalLine.set_color('blue')
        else:
            self.verticalLine = None

        self.visibleBool = True
        axes.legend(loc='upper left')
        self.draw_idle()

    def clear_plot(self):
        self.figure.clf()
        self.draw_idle()

        if self.verticalLine is not None:
            # self.verticalLine is automatically removed by self.figure.clf()
            self.verticalLine = None

        self.visibleBool = False

    def change_vline_position(self, pos):
        self.verticalLinePos = pos

        if self.visibleBool:
            if self.verticalLine is not None:
                self.verticalLine.remove()
                self.verticalLine = None

            axes = self.figure.gca()
            lowerBound, upperBound = axes.get_xlim()

            if lowerBound <= self.verticalLinePos <= upperBound:
                self.verticalLine = axes.axvline(self.verticalLinePos)
                self.verticalLine.set_color('blue')

                self.draw_idle()


    def on_click(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Note: event.x/y = relative position, event.xdata/ydata = absolute position
        xAbsolute, yAbsolute = event.xdata, event.ydata

        action = event.button  # event.button = mouse(1,2,3)

        if action == 1:
            self.callback(xAbsolute)
