# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotNTFCanvasClass.py
# Description:  Extends plotCanvas for the display of NTFs
# ===========================================================================

import matplotlib.figure
import networkx as nx

from .plotCanvasClass import PlotCanvas
from .utilitiesClass import Utilities


# ======================================================================================================================


class PlotNTFCanvas(PlotCanvas):
    def __init__(self, graph, interface, intervalID, stretchFactor, showNoFlowEdges=True, onlyNTF=False):
        self.figure = matplotlib.figure.Figure()
        self.onlyNTF = onlyNTF  # If this is true, then PlotCanvas belongs to Thinflow Computation App
        self.tfType = interface.currentTF
        if nx.number_of_nodes(graph) != 0:
            # Graph is not empty. Otherwise dont do stuff because the call was made by thinFlow_application initialization
            if not self.onlyNTF:
                # We have a regular nashFlow instance
                flowIntervalInstance = interface.gttr('nashFlow').flowIntervals[intervalID][2]
            else:
                # We just have a flowInterval instance
                flowIntervalInstance = interface.gttr('interval')
            self.NTFNodeSpillbackFactorDict = flowIntervalInstance.NTFNodeSpillbackFactorDict if self.tfType == 'spillback' else None
            self.NTFEdgeInflowBoundDict = nx.get_edge_attributes(flowIntervalInstance.shortestPathNetwork, 'inflowBound')\
                if self.tfType == 'spillback' else None
            self.showSpillBackFactor = (self.tfType == 'spillback')
            self.NTFNodeLabelDict = flowIntervalInstance.NTFNodeLabelDict
            self.NTFEdgeFlowDict = flowIntervalInstance.NTFEdgeFlowDict

            self.resettingEdges = flowIntervalInstance.resettingEdges
            self.fullEdges = flowIntervalInstance.fullEdges if self.tfType == 'spillback' else []

        PlotCanvas.__init__(self, graph, interface, stretchFactor=stretchFactor)  # Call parents constructor
        self.network = self.network.copy()  # Copy network to avoid modifying it in other Canvas when deleting/adding zero flow edges
        self.originalNetwork = self.network.copy()
        if not showNoFlowEdges:
            self.change_edge_show_status(showNoFlowEdges)

    def get_additional_node_labels(self):
        """Returns additional node labels"""
        if nx.number_of_nodes(self.network) == 0:
            return {}

        labelDict = {}
        for node in self.network.nodes():
            labelList = []
            val = self.NTFNodeLabelDict[node]
            if val != int(val):
                entry = float("{0:.2f}".format(val))
            else:
                entry = int(val)
            labelList.append(entry)
            if self.showSpillBackFactor:
                val = self.NTFNodeSpillbackFactorDict[node]
                if val < 1:
                    entry = float("{0:.2f}".format(val))
                    labelList.append(entry)
            labelDict[node] = tuple(labelList) if len(labelList) > 1 else labelList[0]
        return labelDict

    def get_edge_labels(self):
        """Returns edge labels"""
        labelDict = {}
        for edge in self.network.edges():
            labelList = []
            # Add inflow bound b^+_e
            if self.showSpillBackFactor:
                if self.NTFEdgeInflowBoundDict[edge] < float('inf'):
                    if self.NTFEdgeInflowBoundDict[edge] != int(self.NTFEdgeInflowBoundDict[edge]):
                        inflowBoundVal = float("{0:.2f}".format(self.NTFEdgeInflowBoundDict[edge]))
                    else:
                        inflowBoundVal = int(self.NTFEdgeInflowBoundDict[edge])
                    labelList.append(inflowBoundVal)

            # Add flow value x'_e
            if self.NTFEdgeFlowDict[edge] != int(self.NTFEdgeFlowDict[edge]):
                flowVal = float("{0:.2f}".format(self.NTFEdgeFlowDict[edge]))
            else:
                flowVal = int(self.NTFEdgeFlowDict[edge])
            labelList.append(flowVal)

            labelDict[edge] = tuple(labelList) if len(labelList) > 1 else labelList[0]
        return labelDict

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
        """
        Move-Mouse-event handling
        :param event: event which is emitted by matplotlib
        """
        if event.xdata is None or event.ydata is None:
            return

        # Note: event.x/y = relative position, event.xdata/ydata = absolute position
        xAbsolute, yAbsolute = event.xdata, event.ydata

        if self.mouseWheelPressed and self.mouseWheelPressedPosition is not None:
            self.mouseWheelPosition = (xAbsolute, yAbsolute)
            self.move()
            self.draw_idle()

    def change_edge_show_status(self, show=True):
        """
        Change whether zero-flow edges are visible or not
        """
        if show:
            self.network = self.originalNetwork.copy()
            # self.network = deepcopy(self.originalNetwork)
        else:
            removedEdges = []
            for edge in self.network.edges():
                if Utilities.is_eq_tol(self.NTFEdgeFlowDict[edge], 0):
                    removedEdges.append(edge)
            self.network.remove_edges_from(removedEdges)
            isolated_nodes = list(nx.isolates(self.network))
            self.network.remove_nodes_from(isolated_nodes)

        self.init_plot()

    def edgeColor(self, v, w):
        """
        Function returning the color that should be used while drawing edges
        :param v: tail node
        :param w: head node
        :return: Color string (e.g. 'b', 'black', 'red' et cetera)
        """
        if (v, w) in self.fullEdges:
            return 'red'
        elif (v, w) in self.resettingEdges:
            return 'blue'
        return 'black'  # Edge should be black if neither resetting nor full

    def get_viewpoint(self):
        """Get the field of view setting"""
        return self.Xlim, self.Ylim, self.edgeWidthSize, self.nodeLabelFontSize, self.edgeLabelFontSize

    def set_viewpoint(self, viewPoint=None):
        """Set the field of view settings"""
        if viewPoint is None:
            return

        self.Xlim, self.Ylim, self.edgeWidthSize, self.nodeLabelFontSize, self.edgeLabelFontSize = viewPoint
        self.zoom(factor=None)
