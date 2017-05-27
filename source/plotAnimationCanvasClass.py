# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotAnimationCanvasClass.py
# ===========================================================================

import networkx as nx
from utilitiesClass import Utilities
import numpy as np
import time
from math import sqrt
import matplotlib.figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import os

from networkx import draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels, draw_networkx_edge_labels

# Config
SIMILARITY_DIST = 9  # Maximal distance at which a click is recognized as a click on a node/edge


# ======================================================================================================================



class PlotAnimationCanvas(FigureCanvas):
    """

    Parameters:
        nashFlow:      nashFlowClass instance
        
    """

    def __init__(self, nashflow, interface, upperBound):
        self.figure = matplotlib.figure.Figure()
        super(PlotAnimationCanvas, self).__init__(self.figure)  # Call parents constructor
        self.nashFlow = nashflow
        self.interface = interface
        self.upperBound = upperBound



        self.network = self.nashFlow.network

        self.currentTimeIndex = 0

        self.precompute_information()

        # Signals
        self.mpl_connect('button_press_event', self.onclick)

        self.focusNode = None
        self.focusEdge = None

    def precompute_information(self):
        self.timePoints = [float(i)/99 * self.upperBound for i in range(100)]
        self.nodeLabelByTimeDict = {node:dict() for node in self.network.nodes()}

        # Node Labels
        for timeIndex in range(100):
            time = self.timePoints[timeIndex]
            for v in self.network.nodes():
                self.nodeLabelByTimeDict[v][time] = self.nashFlow.node_label(v, time)



    def onclick(self, event):
        """
        Onclick-event handling
        :param event: event which is emitted by matplotlib
        """

        if event.xdata is None or event.ydata is None:
            return
        # Note: event.button = mouse(1,2,3), event.x/y = relative position, event.xdata/ydata = absolute position
        xAbsolute, yAbsolute = event.xdata, event.ydata

        # Determine whether we clicked an edge or not
        clickedEdge = self.check_edge_clicked((xAbsolute, yAbsolute))

        # Determine whether we clicked a node or not
        clickedNode = self.check_node_clicked((xAbsolute, yAbsolute), edgePossible=(clickedEdge is not None))

        if clickedEdge is not None and clickedNode is None:
            self.focusEdge = clickedEdge
            self.interface.update_edge_graphs()
        elif clickedNode is not None:
            self.focusNode = clickedNode
            self.interface.update_node_label_graph()

        self.update_plot()


    def check_edge_clicked(self, clickpos):
        """
        Check whether a given click position clickpos was a click on an edge
        :param clickpos: tuple containing absolute x and y value of click event
        :return: clicked edge (None if no edge has been selected)
        """
        clickedEdge = None
        for edge in self.network.edges_iter():
            startpos = self.network.node[edge[0]]['position']
            endpos = self.network.node[edge[1]]['position']
            dist = self.compute_dist_projection_on_segment(clickpos, startpos, endpos)
            if 0 <= dist <= SIMILARITY_DIST:
                clickedEdge = edge
                break
        return clickedEdge

    @staticmethod
    def compute_dist_projection_on_segment(clickpos, startpos, endpos):
        """
        Compute distance between clickpos and its  vertical projection on line segment [startpos, endpos]
        :param clickpos: tuple containing absolute x and y value of click event
        :param startpos: tuple specifying start of line segment
        :param endpos: tuple specifying end of line segment
        :return: distance (-1 if projection does not lie on segment)
        """
        # Subtract startpos to be able to work with vectors
        mu = np.array(clickpos) - np.array(startpos)
        b = np.array(endpos) - np.array(startpos)

        xScalar = (np.dot(mu, b) / (np.linalg.norm(b)) ** 2)
        x = xScalar * b

        if xScalar < 0 or xScalar > 1:
            # The perpendicular projection of mu onto b does not lie on the segment [0, b]
            return -1
        else:
            # Return distance of projection to vector itself
            return np.linalg.norm(mu - x)

    def check_node_clicked(self, clickpos, edgePossible=False):
        """
        Check whether a given click position clickpos was a click on a node
        Creates a node if clickpos is not too close to existing nodes
        :param clickpos: tuple containing absolute x and y value of click event
        :param edgePossible: bool indicating whether click corresponds to an edge
        :return: clicked node (None if no node has been selected)
        """
        xAbsolute, yAbsolute = clickpos[0], clickpos[1]
        clickedNode = None
        minDist = float('inf')
        positions = nx.get_node_attributes(self.network, 'position')
        for node, pos in positions.iteritems():
            dist = sqrt((xAbsolute - pos[0]) ** 2 + (yAbsolute - pos[1]) ** 2)
            if dist <= SIMILARITY_DIST:
                clickedNode = node
                break
            elif dist <= minDist:
                minDist = dist

        return clickedNode

    def update_plot(self):
        """
        Update canvas to plot new graph
        """
        self.figure.clf()  # Clear current figure window
        axes = self.figure.add_axes([0, 0, 1, 1])

        axes.set_xlim(-100, 100)
        axes.set_ylim(-100, 100)
        axes.axis('off')  # Hide axes in the plot

        positions = nx.get_node_attributes(self.network, 'position')
        if self.focusNode is not None:
            draw_networkx_nodes(self.network, pos=positions, ax=axes,
                                nodelist=[self.focusNode], node_color='b')
            remainingNodes = [node for node in self.network.nodes() if node != self.focusNode]
            draw_networkx_nodes(self.network, pos=positions, ax=axes, nodelist=remainingNodes)
        else:
            draw_networkx_nodes(self.network, pos=positions, ax=axes)

        draw_networkx_labels(self.network, pos=positions, ax=axes, labels=nx.get_node_attributes(self.network, 'label'))

        if self.focusEdge is not None:
            draw_networkx_edges(self.network, pos=positions, ax=axes, arrow=True,
                                edgelist=[self.focusEdge], edge_color='b')
            remainingEdges = [edge for edge in self.network.edges() if edge != self.focusEdge]
            draw_networkx_edges(self.network, pos=positions, ax=axes, arrow=True,
                                edgelist=remainingEdges)
        else:
            draw_networkx_edges(self.network, pos=positions, ax=axes, arrow=True)

        # Plot Node Labels
        offset = (0, 8)
        add_tuple_offset = lambda a: (a[0] + offset[0], a[1] + offset[1])
        movedPositions = {edge: add_tuple_offset(positions[edge]) for edge in positions}
        draw_networkx_labels(self.network, pos=movedPositions, ax=axes,
                             labels={node: "%.2f" % self.nodeLabelByTimeDict[node][self.timePoints[self.currentTimeIndex]] for node in self.network.nodes()})


        self.draw_idle()  # Draw only if necessary

    def time_changed(self, sliderVal):
        self.currentTimeIndex = sliderVal
        self.update_plot()