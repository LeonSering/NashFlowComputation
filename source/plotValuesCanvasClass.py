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
    def __init__(self):
        self.figure = figure.Figure()
        super(PlotValuesCanvas, self).__init__(self.figure)  # Call parents constructor

        self.additionalColors = ['red', 'blue']

    def update_plot(self, lowerBound, upperBound, xValues, yValues, *additional_values):
        self.figure.clf()

        axes = self.figure.add_subplot(111)
        yMin, yMax = min(yValues), max(yValues)
        axes.plot(xValues, yValues, linewidth=2, color='green')
        colorCounter = 0
        for xVals, yVals in additional_values:
            yMin, yMax = min(yMin, min(yVals)), max(yMax, max(yVals))
            axes.plot(xVals, yVals, linewidth=2, color=self.additionalColors[colorCounter])
            colorCounter += 1

        axes.set_xlim(lowerBound, upperBound)
        axes.set_ylim(max(0, yMin), int(max(1, yMax) * 1.5))

        self.draw_idle()

    def clear_plot(self):
        self.figure.clf()
        self.draw_idle()
