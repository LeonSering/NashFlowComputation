# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotValuesCanvasClass.py
# Description:  Class to plot diagrams
# ===========================================================================

import matplotlib
from matplotlib import figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .utilitiesClass import Utilities


# ======================================================================================================================


class PlotValuesCanvas(FigureCanvas):
    def __init__(self, callback=None):
        self.figure = figure.Figure()
        super(PlotValuesCanvas, self).__init__(self.figure)  # Call parents constructor

        self.figure.patch.set_facecolor("lightgrey")
        self.plots = []
        self.hLines = []
        self.hLinesLabels = []
        self.legendToLineDict = dict()
        self.lineToHLineDict = dict()
        self.additionalColors = ['green', 'red', 'blue', 'orange']
        self.verticalLine = None
        self.verticalLinePos = 0
        self.verticalLineColor = 'grey'
        self.verticalLineWidth = 0.5
        self.storageHLine = None
        self.storageHLineColor = 'black'
        self.storageHLineWidth = 0.8
        self.callback = callback

        self.visibleBool = False
        self.picked = False

        # Signals
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('pick_event', self.on_click)

        # Enforce using PDFLatex instead of xetex
        pgf_with_pdflatex = {
            "pgf.texsystem": "pdflatex"}
        matplotlib.rcParams.update(pgf_with_pdflatex)

    def update_plot(self, lowerBound, upperBound, labels, xValues, yValues, storage=None, *additional_values):
        """
        Update the plot
        :param lowerBound: lower bound for X range
        :param upperBound: upper bound for X range
        :param labels: legends
        :param xValues: list of X values
        :param yValues: list of Y values
        :param storage: storage value
        :param additional_values: list of tuples: more values to plot
        :return: 
        """
        self.figure.clf()
        self.plots = []
        self.hLines = []
        self.hLinesLabels = []
        self.legendToLineDict = dict()
        self.lineToHLineDict = dict()
        self.storageHLine = None

        axes = self.figure.add_subplot(111)
        lines = []

        yMin, yMax = min(yValues), max(yValues)
        line, = axes.plot(xValues, yValues, linewidth=2, color=self.additionalColors[0], label=labels[0])
        self.plots.append((xValues, yValues, line, labels[0]))
        lines.append(line)
        colorCounter = 1
        for xVals, yVals in additional_values:
            yMin, yMax = min(yMin, min(yVals)), max(yMax, max(yVals))
            line, = axes.plot(xVals, yVals, linewidth=2, color=self.additionalColors[colorCounter],
                              label=labels[colorCounter])
            self.plots.append((xVals, yVals, line, labels[colorCounter]))
            lines.append(line)
            colorCounter += 1

        axes.set_xlim(lowerBound, upperBound)
        axes.set_ylim(max(0, yMin), int(max(1, yMax) * 1.5))

        legend = axes.legend(loc='upper left')

        for idx, legend_entry in enumerate(legend.get_lines()):
            legend_entry.set_picker(5)  # 5 pts tolerance
            self.legendToLineDict[legend_entry] = lines[idx]

        if storage is not None and axes.get_ylim()[0] <= storage <= axes.get_ylim()[1]:
            # Add the storage line
            self.add_storage_hline_to_plot(storage)

        if lowerBound <= self.verticalLinePos <= upperBound:
            self.verticalLine = axes.axvline(self.verticalLinePos, linewidth=self.verticalLineWidth)
            self.verticalLine.set_color(self.verticalLineColor)
            self.add_hline_to_plots()
        else:
            self.verticalLine = None

        self.visibleBool = True
        self.draw_idle()

    def clear_plot(self):
        """Clear the plot"""
        self.figure.clf()
        self.plots = []
        self.hLines = []
        self.hLinesLabels = []
        self.legendToLineDict = dict()
        self.lineToHLineDict = dict()
        self.draw_idle()

        self.verticalLine = None  # self.verticalLine is automatically removed by self.figure.clf()
        self.storageHLine = None

        self.visibleBool = False

    def change_vline_position(self, pos):
        """Change the grey lines position to pos"""
        self.verticalLinePos = pos

        if self.visibleBool:
            if self.verticalLine is not None:
                self.verticalLine.remove()
                self.verticalLine = None

            axes = self.figure.gca()
            lowerBound, upperBound = axes.get_xlim()

            if lowerBound <= self.verticalLinePos <= upperBound:
                self.verticalLine = axes.axvline(self.verticalLinePos, linewidth=self.verticalLineWidth)
                self.verticalLine.set_color(self.verticalLineColor)
                self.add_hline_to_plots()
                self.draw_idle()

    def on_click(self, event):
        """
        Onclick-event handling
        :param event: event which is emitted by matplotlib
        """
        callName = event.name
        # Need to catch pick_event as every pick_event emits another click_event
        if callName == "pick_event":
            self.picked = True
            return self.on_pick(event)
        elif self.picked:
            self.picked = False
            return

        if event.xdata is None or event.ydata is None:
            return
        # Note: event.x/y = relative position, event.xdata/ydata = absolute position
        xAbsolute, yAbsolute = event.xdata, event.ydata

        action = event.button  # event.button = mouse(1,2,3)

        if action == 1:
            self.callback(xAbsolute)

    def on_pick(self, event):
        """
        Onpick-event handling
        :param event: event which is emitted by matplotlib
        """
        legend_entry = event.artist
        line = self.legendToLineDict[legend_entry]
        visibility = not line.get_visible()
        line.set_visible(visibility)

        # Change visibility of hLines if they exist
        try:
            hLine, hLineText = self.lineToHLineDict[line]
            hLine.set_visible(visibility)
            hLineText.set_visible(visibility)
        except KeyError:
            pass

        # Change alpha of legend to indicate visibility
        if visibility:
            legend_entry.set_alpha(1.0)
        else:
            legend_entry.set_alpha(0.2)
        self.draw_idle()

    def export(self, path):
        """Export diagram to file"""
        # Hide Lines for export
        if self.verticalLine is not None:
            self.verticalLine.set_visible(False)
            for l in self.hLines:
                l.set_visible(False)
            for label in self.hLinesLabels:
                label.set_visible(False)

        self.draw_idle()
        self.figure.savefig(path)

        # Show Lines again
        if self.verticalLine is not None:
            self.verticalLine.set_visible(True)
            for l in self.hLines:
                l.set_visible(True)
            for label in self.hLinesLabels:
                label.set_visible(True)
        self.draw_idle()

    def add_storage_hline_to_plot(self, storage):
        """Add storage horizontal line to the plot"""
        axes = self.figure.gca()
        self.storageHLine = axes.axhline(y=storage, linewidth=self.storageHLineWidth)
        self.storageHLine.set_color(self.storageHLineColor)

    def add_hline_to_plots(self):
        """Add horizontal lines to intersect vertical line"""
        for hLine in self.hLines:
            hLine.remove()
        self.hLines = []
        self.lineToHLineDict = dict()

        for label in self.hLinesLabels:
            label.remove()
        self.hLinesLabels = []

        for plot in self.plots:
            xVals, yVals, line, label = plot

            # Get the y-value of self.verticalLinePos
            index = Utilities.get_insertion_point_left(xVals, self.verticalLinePos)
            if index == len(xVals) or index == 0:
                continue
            x1, x, x2 = xVals[index - 1], self.verticalLinePos, xVals[index]
            y1, y2 = yVals[index - 1], yVals[index]
            # It holds xVals[index-1] < self.verticalLinePos <= xVals[index]
            fac = float(x - x2) / (x1 - x2)
            y = fac * y1 + (1 - fac) * y2  # this obviously only works if plots are piecewise linear

            axes = self.figure.gca()
            lowerBound, upperBound = axes.get_xlim()
            lineBeginFac = float(x - lowerBound) / (upperBound - lowerBound)
            hLine = axes.axhline(y=y, xmin=lineBeginFac, xmax=1, linewidth=self.verticalLineWidth)
            hLine.set_color(self.verticalLineColor)
            self.hLines.append(hLine)
            hLineText = axes.text(1.02, y, "%.2f" % y, va='center', ha="left", bbox=dict(facecolor="w", alpha=0.5),
                                  transform=axes.get_yaxis_transform())
            hLineText.set_fontsize(8)
            self.hLinesLabels.append(hLineText)

            if not line.get_visible():
                hLine.set_visible(False)
                hLineText.set_visible(False)
            self.lineToHLineDict[line] = (hLine, hLineText)

    def get_approx_load(self, t):
        """
        Returns approximated load at time t
        :param t:
        :return: d_e(t) or N/A
        """
        for plot in self.plots:
            xVals, yVals, line, label = plot
            if label != 'Load':
                continue

            # Get the y-value of self.verticalLinePos
            index = Utilities.get_insertion_point_left(xVals, t)
            if index == len(xVals) or index == 0:
                return "N/A"
            x1, x, x2 = xVals[index - 1], t, xVals[index]
            y1, y2 = yVals[index - 1], yVals[index]
            # It holds xVals[index-1] < t <= xVals[index]
            fac = float(x - x2) / (x1 - x2)
            y = fac * y1 + (1 - fac) * y2  # this obviously only works if plots are piecewise linear
            return y

        return "N/A"