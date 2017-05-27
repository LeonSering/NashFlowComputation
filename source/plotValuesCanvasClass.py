# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotValuesCanvasClass.py
# ===========================================================================

from matplotlib import figure, lines
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas



# ======================================================================================================================



class PlotValuesCanvas(FigureCanvas):
    def __init__(self):
        self.figure = figure.Figure()
        super(PlotValuesCanvas, self).__init__(self.figure)  # Call parents constructor


    def update_plot(self, lowerBound, upperBound, xValues, yValues, *additional_yValues):
        self.figure.clf()

        axes = self.figure.add_subplot(111)
        #axes.set_xlabel('time')
        #axes.set_ylabel('raw data')

        axes.plot(xValues, yValues, linewidth=2, color='blue')
        for arg in additional_yValues:
            axes.plot(xValues, arg, linewidth=2, color='red')

        axes.set_xlim(lowerBound, upperBound)
        axes.set_ylim(min(yValues), max(yValues))


        self.draw_idle()