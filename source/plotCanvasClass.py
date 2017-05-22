# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotCanvasClass.py
# Description:  Class to plot networkx graphs in widgets and control click events on said graphs
# Parameters:   figure:     matplotlib.figure.Figure instance
#               graph:      CurrentGraph instance
#               interface:  Interface instance
# ===========================================================================

import numpy as np
import time
from math import sqrt
import matplotlib.figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import os

from networkx import draw_networkx_nodes, draw_networkx_edges, draw_networkx_labels, draw_networkx_edge_labels


# Config
SIMILARITY_DIST = 9  # Maximal distance at which a click is recognized as a click on a node/edge
TIME_DIFF = 0.3  # Minimum time between mouse press and release s.t. it is recognized as "drag and drop" (in seconds)


# ======================================================================================================================



class PlotCanvas(FigureCanvas):
    """
    Class to plot networkx graphs in widgets and control click events on said graphs
    Parameters:
        figure:     matplotlib.figure.Figure instance
        graph:      CurrentGraph instance
        interface:  Interface instance
    """

    def __init__(self, graph, interface):
        self.figure = matplotlib.figure.Figure()
        super(PlotCanvas, self).__init__(self.figure)  # Call parents constructor

        self.currentGraph = graph
        self.interface = interface

        # Signals
        self.mpl_connect('button_press_event', self.onclick)
        self.mpl_connect('button_release_event', self.onrelease)

        # Mouse events
        self.mouseReleased = False
        self.mousePressed = False
        self.mouseReleaseTime = None
        self.mousePressTime = None
        self.pressedNode = None
        self.releasedNode = None

        self.focusNode = None
        self.focusEdge = None

    def onclick(self, event):
        """
        Onclick-event handling
        :param event: event which is emitted by matplotlib
        """
        self.mousePressed = True
        self.mousePressTime = time.time()

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
            self.interface.update_edge_display()
        elif clickedNode is not None:
            self.pressedNode = clickedNode
            self.focusNode = clickedNode
            self.interface.update_node_display()

    def onrelease(self, event):
        """
        Release-Mouse-event handling
        :param event: event which is emitted by matplotlib
        """
        self.mouseReleased = True
        self.mouseReleaseTime = time.time()
        self.update_plot()

        if self.mouseReleaseTime - self.mousePressTime < TIME_DIFF:
            return
        if event.xdata is None or event.ydata is None:
            return
        xAbsolute, yAbsolute = event.xdata, event.ydata

        # Determine whether we clicked a node or not
        clickedNode = self.check_node_clicked((xAbsolute, yAbsolute), edgePossible=True)

        if clickedNode is not None:
            self.releasedNode = clickedNode
            if self.pressedNode is not None and self.pressedNode != self.releasedNode:
                # Add the corresponding edge, if valid
                if not self.currentGraph.has_edge(self.pressedNode, self.releasedNode):
                    self.currentGraph.add_edge(self.pressedNode, self.releasedNode, transitTime=0, capacity=0)
                    self.focusEdge = (self.pressedNode, self.releasedNode)
                    self.interface.update_edge_display()
                    self.update_plot()

        self.mousePressed = False
        self.mouseReleased = False

        self.pressedNode = None
        self.releasedNode = None

        self.mouseReleaseTime = None
        self.mousePressTime = None

    def check_edge_clicked(self, clickpos):
        """
        Check whether a given click position clickpos was a click on an edge
        :param clickpos: tuple containing absolute x and y value of click event
        :return: clicked edge (None if no edge has been selected)
        """
        clickedEdge = None
        for edge in self.currentGraph.edges_iter():
            startpos = self.currentGraph.position[edge[0]]
            endpos = self.currentGraph.position[edge[1]]
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
        for node, pos in self.currentGraph.position.iteritems():
            dist = sqrt((xAbsolute - pos[0]) ** 2 + (yAbsolute - pos[1]) ** 2)
            if dist <= SIMILARITY_DIST:
                clickedNode = node
                break
            elif dist <= minDist:
                minDist = dist

        if clickedNode is None and minDist > 2 * SIMILARITY_DIST and not edgePossible:
            # Create new node
            nodeID = str(self.currentGraph.lastID)
            self.currentGraph.add_node(nodeID)
            self.currentGraph.position[nodeID] = (int(xAbsolute), int(yAbsolute))
            self.currentGraph.label[nodeID] = nodeID
            self.currentGraph.lastID += 1
            return nodeID
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

        if self.focusNode is not None:
            draw_networkx_nodes(self.currentGraph, pos=self.currentGraph.position, ax=axes,
                                nodelist=[self.focusNode], node_color='b')
            remainingNodes = [node for node in self.currentGraph.nodes() if node != self.focusNode]
            draw_networkx_nodes(self.currentGraph, pos=self.currentGraph.position, ax=axes, nodelist=remainingNodes)
        else:
            draw_networkx_nodes(self.currentGraph, pos=self.currentGraph.position, ax=axes)
        draw_networkx_labels(self.currentGraph, pos=self.currentGraph.position, ax=axes, labels=self.currentGraph.label)

        if self.focusEdge is not None:
            draw_networkx_edges(self.currentGraph, pos=self.currentGraph.position, ax=axes, arrow=True,
                                edgelist=[self.focusEdge], edge_color='b')
            remainingEdges = [edge for edge in self.currentGraph.edges() if edge != self.focusEdge]
            draw_networkx_edges(self.currentGraph, pos=self.currentGraph.position, ax=axes, arrow=True,
                                edgelist=remainingEdges)
        else:
            draw_networkx_edges(self.currentGraph, pos=self.currentGraph.position, ax=axes, arrow=True)

        lbls = {
            edge: (self.currentGraph[edge[0]][edge[1]]['transitTime'], self.currentGraph[edge[0]][edge[1]]['capacity'])
            for edge in self.currentGraph.edges()
        }  # Edge labels
        draw_networkx_edge_labels(self.currentGraph, pos=self.currentGraph.position, ax=axes, edge_labels=lbls)

        self.draw_idle()  # Draw only if necessary
        