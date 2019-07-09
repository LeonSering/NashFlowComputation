# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotNTFCanvasClass.py
# Description:  Extends plotCanvas for the display of NTFs
# ===========================================================================

import matplotlib.figure

import networkx as nx

from source.plotCanvasClass import PlotCanvas
from source.utilitiesClass import Utilities

# ======================================================================================================================


class PlotNTFCanvas(PlotCanvas):
    def __init__(self, graph, interface, intervalID, stretchFactor, showNoFlowEdges=True, onlyNTF=False):
        self.figure = matplotlib.figure.Figure()
        self.onlyNTF = onlyNTF  # If this is true, then PlotCanvas belongs to Thinflow Computation App
        self.tfType = 'general'
        if nx.number_of_nodes(graph) != 0:
            # Graph is not empty. Otherwise dont do stuff because the call was made by thinFlow_application initialization
            if not self.onlyNTF:
                # We have a regular nashFlow instance
                flowIntervalInstance = interface.gttr('nashFlow').flowIntervals[intervalID][2]
                self.showSpillBackFactor = False
            else:
                # We just have a flowInterval instance
                self.tfType = interface.currentTF
                flowIntervalInstance = interface.gttr('interval')
                self.NTFNodeSpillbackFactorDict = flowIntervalInstance.NTFNodeSpillbackFactorDict if self.tfType == 'spillback' else None
                self.showSpillBackFactor = (self.onlyNTF and self.tfType == 'spillback')
            self.NTFNodeLabelDict = flowIntervalInstance.NTFNodeLabelDict
            self.NTFEdgeFlowDict = flowIntervalInstance.NTFEdgeFlowDict

            self.resettingEdges = flowIntervalInstance.resettingEdges

        PlotCanvas.__init__(self, graph, interface, stretchFactor=stretchFactor)  # Call parents constructor
        self.network = self.network.copy()  # Copy network to avoid modifying it in other Canvas when deleting/adding zero flow edges
        self.originalNetwork = self.network.copy()
        if not showNoFlowEdges:
            self.change_edge_show_status(showNoFlowEdges)

    def get_additional_node_labels(self):
        """Returns additional node labels"""
        if nx.number_of_nodes(self.network) == 0:
            return {}
        if self.showSpillBackFactor:
            return {
                node: (
                    float("%.2f" % self.NTFNodeLabelDict[node]), float("%.2f" % self.NTFNodeSpillbackFactorDict[node]))
                for
                node in self.network.nodes()}
        else:
            return {node: "%.2f" % self.NTFNodeLabelDict[node] for
                    node in self.network.nodes()}

    def get_edge_labels(self):
        """Returns edge labels"""
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
        resettingFlag = False
        fullFlag = False
        '''
        if (v, w) in self.resettingEdges:
            # Resetting edge
            resettingFlag = True
        if Utilities.is_geq_tol(self.NTFEdgeFlowDict[(v,w)], self.network[v][w]['capacity']):
            fullFlag = True
        if resettingFlag and fullFlag:
            # Mix colors -> orange
            return '#FF8000'
        elif resettingFlag:
            return 'red'
        elif fullFlag:
            return '#FFFF00'   # Yellow
        '''
        if (v, w) in self.resettingEdges:
            return 'red'
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
