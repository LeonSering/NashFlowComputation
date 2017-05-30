# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotCanvasClass.py
# Description:  Class to plot networkx graphs in widgets and control click events on said graphs
# Parameters:   graph:      nx.Digraph instance
#               interface:  Interface instance
#               clickable:  (bool) if True then canvas is clickable (i.e. drag&drop, selection, etc)
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
DRAG_DROP_TIME_DIFF = 0.3  # Minimum time between mouse press and release s.t. it is recognized as "drag and drop" (in seconds)
MOVE_AXIS_TIME_DIFF = 0.15


# ======================================================================================================================



class PlotCanvas(FigureCanvas):
    """
    Class to plot networkx graphs in widgets and control click events on said graphs
    Parameters:
        graph:      nx.Digraph instance
        interface:  Interface instance
        clickable:  (bool) if True then canvas is clickable (i.e. drag&drop, selection, etc)
    """

    def __init__(self, graph, interface, clickable=True, intervalID=None):
        self.figure = matplotlib.figure.Figure()
        super(PlotCanvas, self).__init__(self.figure)  # Call parents constructor

        self.network = graph
        self.interface = interface
        self.displaysNTF = (not clickable)

        # Visualization Settings
        self.Xlim = (-100, 100)
        self.Ylim = (-100, 100)
        self.nodeSize = 300
        self.nodeLabelFontSize = 12 # float but passed as int
        self.edgeLabelFontSize = 10 # float but passed as int


        if not self.displaysNTF:
            # Signals
            self.mpl_connect('button_press_event', self.onclick)
            self.mpl_connect('button_release_event', self.onrelease)
        else:
            flowLabels = self.interface.nashFlow.flowIntervals[intervalID][2].NTFEdgeFlowDict
            self.NTFEdgeFlowDict = {edge:flowLabels[edge] for edge in self.network.edges()}
            self.NTFNodeLabelDict = self.interface.nashFlow.flowIntervals[intervalID][2].NTFNodeLabelDict

        self.mpl_connect('scroll_event', self.onscroll)

        # Mouse events
        self.mouseReleased = False
        self.mousePressed = False
        self.mouseReleaseTime = None
        self.mousePressTime = None
        self.pressedNode = None
        self.releasedNode = None
        self.mouseWheelPressedPosition = None
        self.mouseWheelPressedTime = None
        self.mouseWheelReleasedPosition = None
        self.mouseWheelReleasedTime = None

        self.focusNode = None
        self.focusEdge = None

    def onclick(self, event):
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
            # Wheel was clicked, prepare to move visible part of canvas
            self.mouseWheelPressedPosition = (xAbsolute, yAbsolute)
            self.mouseWheelPressedTime = time.time()
            return

        self.mousePressed = True
        self.mousePressTime = time.time()

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

        if event.xdata is None or event.ydata is None:
            return
        xAbsolute, yAbsolute = event.xdata, event.ydata

        action = event.button   # event.button = mouse(1,2,3)
        if action == 2:
            # Wheel was released
            self.mouseWheelReleasedPosition = (xAbsolute, yAbsolute)
            self.mouseWheelReleasedTime = time.time()

            if self.mouseWheelReleasedTime is not None and self.mouseWheelPressedTime is not None:
                if self.mouseWheelReleasedTime - self.mouseWheelReleasedTime < MOVE_AXIS_TIME_DIFF:
                    self.move()
            return

        self.mouseReleased = True
        self.mouseReleaseTime = time.time()


        if self.mouseReleaseTime - self.mousePressTime < DRAG_DROP_TIME_DIFF:
            self.update_plot()
            return






        # Determine whether we clicked a node or not
        clickedNode = self.check_node_clicked((xAbsolute, yAbsolute), edgePossible=True)

        if clickedNode is not None:
            self.releasedNode = clickedNode
            if self.pressedNode is not None and self.pressedNode != self.releasedNode:
                # Add the corresponding edge, if valid
                if not self.network.has_edge(self.pressedNode, self.releasedNode):
                    self.network.add_edge(self.pressedNode, self.releasedNode, transitTime=0, capacity=0)
                    self.focusEdge = (self.pressedNode, self.releasedNode)
                    self.interface.update_edge_display()

        elif self.pressedNode is not None and self.pressedNode not in ['s', 't'] and not self.check_edge_clicked((xAbsolute, yAbsolute)):
            # Move the node
            self.network.node[self.pressedNode]['position'] = (xAbsolute, yAbsolute)
            self.focusNode = self.pressedNode
            self.interface.update_node_display()


        self.update_plot()

        self.mousePressed = False
        self.mouseReleased = False

        self.pressedNode = None
        self.releasedNode = None

        self.mouseReleaseTime = None
        self.mousePressTime = None

    def onscroll(self, event):
        if event.xdata is None or event.ydata is None:
            return

        xAbsolute, yAbsolute = event.xdata, event.ydata
        action = event.button   # 'up'/'down' Note: 'up' == zoom in,'down' == zoom out
        sFactor = 1-0.1         # zoom out velocity
        bFactor = 1./sFactor    # zoom in velocity, chosen s.t. sFactor * bFactor ~=ye 1

        factor = bFactor if action == 'up' else sFactor
        self.zoom(factor)


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

        if clickedNode is None and minDist > 2 * SIMILARITY_DIST and not edgePossible:
            # Create new node
            nodeID = str(self.network.graph['lastID'])
            self.network.add_node(nodeID)
            self.network.node[nodeID]['position'] = (int(xAbsolute), int(yAbsolute))
            self.network.node[nodeID]['label'] = nodeID
            self.network.graph['lastID'] += 1
            return nodeID
        return clickedNode

    def update_plot(self):
        """
        Update canvas to plot new graph
        """
        self.figure.clf()  # Clear current figure window
        axes = self.figure.add_axes([0, 0, 1, 1])


        axes.set_xlim(self.Xlim)
        axes.set_ylim(self.Ylim)
        axes.axis('off')  # Hide axes in the plot

        nodeLabelSize = int(round(self.nodeLabelFontSize))
        edgeLabelSize = int(round(self.edgeLabelFontSize))

        positions = nx.get_node_attributes(self.network, 'position')
        if self.focusNode is not None:
            draw_networkx_nodes(self.network, pos=positions, ax=axes,
                                nodelist=[self.focusNode], node_color='b', node_size=self.nodeSize)
            remainingNodes = [node for node in self.network.nodes() if node != self.focusNode]
            draw_networkx_nodes(self.network, pos=positions, ax=axes, nodelist=remainingNodes, node_size=self.nodeSize)
        else:
            draw_networkx_nodes(self.network, pos=positions, ax=axes, node_size=self.nodeSize)

        draw_networkx_labels(self.network, pos=positions, ax=axes, labels=nx.get_node_attributes(self.network, 'label'), font_size=nodeLabelSize)

        if self.focusEdge is not None:
            draw_networkx_edges(self.network, pos=positions, ax=axes, arrow=True,
                                edgelist=[self.focusEdge], edge_color='b')
            remainingEdges = [edge for edge in self.network.edges() if edge != self.focusEdge]
            draw_networkx_edges(self.network, pos=positions, ax=axes, arrow=True,
                                edgelist=remainingEdges)
        else:
            draw_networkx_edges(self.network, pos=positions, ax=axes, arrow=True)


        if not self.displaysNTF:
            # Plot actual edge labels
            lbls = Utilities.join_dicts(nx.get_edge_attributes(self.network, 'capacity'),
                                        nx.get_edge_attributes(self.network, 'transitTime'))  # Edge labels
        else:
            # Plot flow value
            lbls = {edge:"%.2f" % self.NTFEdgeFlowDict[edge] for edge in self.NTFEdgeFlowDict}

            # Plot NTFNodeLabels
            offset = (0, 8)
            add_tuple_offset = lambda a: (a[0] + offset[0], a[1] + offset[1])
            movedPositions = {edge:add_tuple_offset(positions[edge]) for edge in positions}
            draw_networkx_labels(self.network, pos=movedPositions, ax=axes,
                                 labels={node:"%.2f" % self.NTFNodeLabelDict[node] for node in self.NTFNodeLabelDict}, font_size=nodeLabelSize)




        draw_networkx_edge_labels(self.network, pos=positions, ax=axes, edge_labels=lbls, font_size=edgeLabelSize)
        self.draw_idle()  # Draw only if necessary

    def zoom(self, factor):

        smaller = lambda val: factor*val         # Returns smaller value if factor < 1, i.e. if zooming out
        bigger = lambda val: (1./factor)*val     # Returns bigger value if factor < 1, i.e. if zooming out

        # Scale axis
        self.Xlim = tuple(bigger(entry) for entry in self.Xlim)
        self.Ylim = tuple(bigger(entry) for entry in self.Ylim)

        # Scale node size
        self.nodeSize = smaller(self.nodeSize)

        # Scale font size of node labels
        self.nodeLabelFontSize = smaller(self.nodeLabelFontSize)

        # Scale font size of edge labels
        self.edgeLabelFontSize = smaller(self.edgeLabelFontSize)

        self.update_plot()

    def move(self):
        dx = self.mouseWheelReleasedPosition[0] - self.mouseWheelPressedPosition[0]
        dy = self.mouseWheelReleasedPosition[1] - self.mouseWheelPressedPosition[1]

        self.Xlim = tuple(entry - dx for entry in self.Xlim)
        self.Ylim = tuple(entry - dy for entry in self.Ylim)

        self.update_plot()