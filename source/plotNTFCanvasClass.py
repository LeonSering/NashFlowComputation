# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotNTFCanvasClass.py
# Description:
# Parameters:
# ===========================================================================

import matplotlib.figure
from utilitiesClass import Utilities
from plotCanvasClass import PlotCanvas
import networkx as nx
matplotlib.use("Qt4Agg")


# ======================================================================================================================



class PlotNTFCanvas(PlotCanvas):
    def __init__(self, graph, interface, intervalID, stretchFactor, showNoFlowEdges=True):
        self.figure = matplotlib.figure.Figure()

        flowLabels = interface.nashFlow.flowIntervals[intervalID][2].NTFEdgeFlowDict
        self.NTFNodeLabelDict = interface.nashFlow.flowIntervals[intervalID][2].NTFNodeLabelDict
        self.NTFEdgeFlowDict = {edge: flowLabels[edge] for edge in graph.edges()}

        PlotCanvas.__init__(self, graph, interface, stretchFactor=stretchFactor)  # Call parents constructor
        self.network = self.network.copy() # Copy network to avoid modifying it in other Canvas when deleting/adding zero flow edges
        self.originalNetwork = self.network.copy()
        if not showNoFlowEdges:
            self.change_edge_show_status(showNoFlowEdges)

    def get_additional_node_labels(self):
        return {node: "%.2f" % self.NTFNodeLabelDict[node] for node in self.network.nodes()}

    def get_edge_labels(self):
        return {edge: "%.2f" % self.NTFEdgeFlowDict[edge] for edge in self.network.edges()}

    def on_click(self, event):
        """
        Onclick-event handling
        :param event: event which is emitted by matplotlib
        """

        if event.xdata is None or event.ydata is None:
            return

        # Note: event.x/y = relative position, event.xdata/ydata = absolute position
        xAbsolute, yAbsolute = event.xdata, event.ydata

        action = event.button  # event.button = mouse(1,2,3)

        if action == 2:
            # Wheel was clicked, move visible part of canvas
            self.mouseWheelPressed = True
            self.mouseWheelPressedPosition = (xAbsolute, yAbsolute)

    def on_release(self, event):
        """
        Release-Mouse-event handling
        :param event: event which is emitted by matplotlib
        """
        if event.xdata is None or event.ydata is None:
            return
        action = event.button  # event.button = mouse(1,2,3)

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

    def change_edge_show_status(self, show=True):
        if show:
            self.network = self.originalNetwork.copy()
        else:
            for edge in self.network.edges():
                if Utilities.is_eq_tol(self.NTFEdgeFlowDict[edge], 0):
                    self.network.remove_edge(edge[0], edge[1])
            self.network.remove_nodes_from(nx.isolates(self.network))

        self.init_plot()

    def get_viewpoint(self):
        return (self.Xlim, self.Ylim, self.edgeWidthSize, self.nodeLabelFontSize, self.edgeLabelFontSize)

    def set_viewpoint(self, viewPoint=None):
        if viewPoint is None:
            return

        self.Xlim, self.Ylim, self.edgeWidthSize, self.nodeLabelFontSize, self.edgeLabelFontSize = viewPoint

        self.zoom(factor=None)

