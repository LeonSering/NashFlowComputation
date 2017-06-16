# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotNTFCanvasClass.py
# Description:
# Parameters:
# ===========================================================================

from plotCanvasClass import PlotCanvas
import time
import matplotlib.figure
matplotlib.use("Qt4Agg")

# ======================================================================================================================



class PlotNTFCanvas(PlotCanvas):

    def __init__(self, graph, interface, intervalID):
        self.figure = matplotlib.figure.Figure()

        flowLabels = interface.nashFlow.flowIntervals[intervalID][2].NTFEdgeFlowDict
        self.NTFNodeLabelDict = interface.nashFlow.flowIntervals[intervalID][2].NTFNodeLabelDict
        self.NTFEdgeFlowDict = {edge:flowLabels[edge] for edge in graph.edges()}


        PlotCanvas.__init__(self, graph, interface) # Call parents constructor

    def get_additional_node_labels(self):
        return {node:"%.2f" % self.NTFNodeLabelDict[node] for node in self.NTFNodeLabelDict}

    def get_edge_labels(self):
        return {edge:"%.2f" % self.NTFEdgeFlowDict[edge] for edge in self.NTFEdgeFlowDict}

    def on_click(self, event):
        """
        Onclick-event handling
        :param event: event which is emitted by matplotlib
        """

        if event.xdata is None or event.ydata is None:
            return

        # Note: event.x/y = relative position, event.xdata/ydata = absolute position
        xAbsolute, yAbsolute = event.xdata, event.ydata

        action = event.button   # event.button = mouse(1,2,3)

        if action == 2:
            # Wheel was clicked, move visible part of canvas
            self.currentXlim = self.Xlim
            self.currentYlim = self.Ylim
            self.mouseWheelPressed = True
            self.mouseWheelPressedPosition = (xAbsolute, yAbsolute)


    def on_release(self, event):
        """
        Release-Mouse-event handling
        :param event: event which is emitted by matplotlib
        """
        if event.xdata is None or event.ydata is None:
            return
        xAbsolute, yAbsolute = event.xdata, event.ydata
        action = event.button   # event.button = mouse(1,2,3)


        if action == 2:
            # Wheel has been released
            self.mouseWheelPressed = False
            self.mouseWheelPressedPosition = None


    def on_motion(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Note: event.x/y = relative position, event.xdata/ydata = absolute position
        xAbsolute, yAbsolute = event.xdata, event.ydata

        if self.mouseWheelPressed and self.mouseWheelPressedPosition is not None:
            self.mouseWheelPosition = (xAbsolute, yAbsolute)
            self.move()
            self.draw_idle()

