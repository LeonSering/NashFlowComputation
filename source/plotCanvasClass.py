# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotCanvasClass.py
# Description:  Class to plot networkx graphs in widgets and control click events on said graphs
# Parameters:   graph:      nx.Digraph instance
#               interface:  Interface instance
#               clickable:  (bool) if True then canvas is clickable (i.e. drag&drop, selection, etc)
# ===========================================================================

import matplotlib.figure
import numpy as np
import time
from math import sqrt

import networkx as nx

from utilitiesClass import Utilities

matplotlib.use("Qt4Agg")
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import colorConverter

from networkx import draw_networkx_nodes, draw_networkx_labels, draw_networkx_edge_labels

# Config
SIMILARITY_DIST = 9  # Maximal distance at which a click is recognized as a click on a node/edge
DRAG_DROP_TIME_DIFF = 0.3  # Minimum time between mouse press and release s.t. it is recognized as "drag and drop" (in seconds)


# ======================================================================================================================



class PlotCanvas(FigureCanvas):
    """
    Class to plot networkx graphs in widgets and control click events on said graphs
    Parameters:
        graph:      nx.Digraph instance
        interface:  Interface instance
        creationBool:  (bool) if True then canvas can be used to create new nodes/edges (i.e. drag&drop, selection, etc)
    """

    def __init__(self, graph, interface):
        self.figure = matplotlib.figure.Figure()
        super(PlotCanvas, self).__init__(self.figure)  # Call parents constructor

        self.network = graph
        self.interface = interface
        # self.displaysNTF = (not clickableBool and not creationBool)
        # self.displaysNashFlow = (clickableBool and not creationBool)

        # Visualization Settings
        self.Xlim = (-100, 100)
        self.Ylim = (-100, 100)
        self.nodeSize = 300
        self.nodeLabelFontSize = 12  # float but passed as int
        self.edgeLabelFontSize = 10  # float but passed as int
        self.focusNode = None
        self.focusEdge = None

        # Internal variables
        self.selectedNode = None


        # Signals
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.mpl_connect('scroll_event', self.on_scroll)

        # Mouse events
        # Left mouse
        self.mouseLeftPressTime = None
        self.mouseLeftReleaseTime = None

        # Mouse wheel
        self.mouseWheelPressedPosition = None
        self.mouseWheelPressed = False

        # Mouse right
        self.mouseRightPressed = False

        self.init_plot()  # Plot for the first time

    def get_additional_node_labels(self):
        return {}

    def get_edge_labels(self):
        return Utilities.join_intersect_dicts(nx.get_edge_attributes(self.network, 'capacity'),
                                              nx.get_edge_attributes(self.network, 'transitTime'))  # Edge labels

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

        if action == 1:
            # Leftmouse was clicked, select/create node, select edge or add edge via drag&drop
            self.mouseLeftPressTime = time.time()

            lastID = self.network.graph['lastID']

            # Determine whether we clicked an edge or not
            clickedEdge = self.check_edge_clicked((xAbsolute, yAbsolute))

            # Determine whether we clicked a node or not
            clickedNode = self.check_node_clicked((xAbsolute, yAbsolute), edgePossible=(clickedEdge is not None))
            newNodeCreated = (self.network.graph['lastID'] > lastID)

            if clickedEdge is not None and clickedNode is None:
                # Selected an existing edge
                self.focusEdge = clickedEdge
                self.interface.update_edge_display()
                self.update_edges(color=True)
            elif clickedNode is not None:
                self.focusNode = clickedNode
                if not newNodeCreated:
                    self.selectedNode = clickedNode
                else:
                    self.mouseLeftPressTime = None
                    self.update_nodes(added=True, color=True)

        elif action == 2:
            # Wheel was clicked, move visible part of canvas
            self.mouseWheelPressed = True
            self.mouseWheelPressedPosition = (xAbsolute, yAbsolute)
            return

        elif action == 3:
            clickedNode = self.check_node_clicked((xAbsolute, yAbsolute), edgePossible=True)
            if clickedNode is not None and clickedNode not in ['s', 't']:
                self.selectedNode = clickedNode
                self.mouseRightPressed = True
                self.focusNode = self.selectedNode
                self.update_nodes(color=True)
                self.interface.update_node_display()

    def on_release(self, event):
        """
        Release-Mouse-event handling
        :param event: event which is emitted by matplotlib
        """
        if event.xdata is None or event.ydata is None:
            return
        xAbsolute, yAbsolute = event.xdata, event.ydata
        action = event.button  # event.button = mouse(1,2,3)

        '''
        if self.displaysNTF and action != 2:
            return
        '''

        if action == 1:
            # Leftmouse has been released
            if not self.mouseLeftPressTime:
                # Released too fast or node has been created
                self.selectedNode = None
            else:
                self.mouseLeftReleaseTime = time.time()
                dtime = self.mouseLeftReleaseTime - self.mouseLeftPressTime

                if dtime < DRAG_DROP_TIME_DIFF:
                    # Time to short for Drag&Drop, just update_plot to show focusNode/focusEdge
                    self.update_nodes(color=True)
                    self.selectedNode = None
                    self.mouseLeftPressTime = None
                    self.mouseLeftReleaseTime = None

                else:
                    # Determine whether we clicked a node or not
                    clickedNode = self.check_node_clicked((xAbsolute, yAbsolute), edgePossible=True)
                    if clickedNode is not None:
                        if self.selectedNode is not None and self.selectedNode != clickedNode:
                            # Add the corresponding edge, if valid
                            if not self.network.has_edge(self.selectedNode, clickedNode):
                                self.network.add_edge(self.selectedNode, clickedNode, transitTime=0, capacity=0)

                                self.focusEdge = (self.selectedNode, clickedNode)
                                self.focusNode = clickedNode

                                self.interface.update_edge_display()
                                self.update_edges(added=True, color=True)
                            self.selectedNode = None

                            # self.update_edges(color=True)

            self.update_nodes(color=True)
            self.interface.update_node_display()

        elif action == 2:
            # Wheel has been released
            self.mouseWheelPressed = False
            self.mouseWheelPressedPosition = None

        elif action == 3:
            # Right mouse has been released
            self.mouseRightPressed = False
            self.selectedNode = None

    def on_motion(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Note: event.x/y = relative position, event.xdata/ydata = absolute position
        xAbsolute, yAbsolute = event.xdata, event.ydata

        if self.mouseWheelPressed and self.mouseWheelPressedPosition is not None:
            self.mouseWheelPosition = (xAbsolute, yAbsolute)
            self.move()
            self.draw_idle()

        elif self.mouseRightPressed and self.selectedNode is not None:
            self.network.node[self.selectedNode]['position'] = (xAbsolute, yAbsolute)
            self.update_nodes(moved=True)
            self.update_edges(moved=True)

    def on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return

        action = event.button  # 'up'/'down' Note: 'up' == zoom in,'down' == zoom out
        sFactor = 1 - 0.1  # zoom out velocity
        bFactor = 1. / sFactor  # zoom in velocity, chosen s.t. sFactor * bFactor ~=ye 1

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

    def init_plot(self):
        """
        Update canvas to plot new graph
        """
        self.figure.clf()  # Clear current figure window
        self.axes = self.figure.add_axes([0, 0, 1, 1])

        self.axes.set_xlim(self.Xlim)
        self.axes.set_ylim(self.Ylim)
        self.axes.axis('off')  # Hide axes in the plot

        nodeLabelSize = int(round(self.nodeLabelFontSize))
        edgeLabelSize = int(round(self.edgeLabelFontSize))

        positions = nx.get_node_attributes(self.network, 'position')

        # Plot Nodes
        self.nodeCollections = []
        nodeColor = lambda v: 'r' if v != self.focusNode else 'b'
        nodeColorList = [nodeColor(v) for v in self.network.nodes()]
        self.nodeCollections.append((self.network.nodes(),
                                     draw_networkx_nodes(self.network, pos=positions, ax=self.axes,
                                                         node_size=self.nodeSize, node_color=nodeColorList)))

        # Plot Node Labels
        self.nodeLabelCollection = draw_networkx_labels(self.network, pos=positions, ax=self.axes,
                                                        labels=nx.get_node_attributes(self.network, 'label'),
                                                        font_size=nodeLabelSize)

        # Plot Edges
        self.edgeCollections, self.arrowCollections = [], []
        edgeColor = lambda v, w: 'black' if (v, w) != self.focusEdge else 'b'
        edgeColorList = [edgeColor(v, w) for v, w in self.network.edges()]
        if edgeColorList:
            edgeCollection, arrowCollection = self.draw_edges(self.network, pos=positions, ax=self.axes,
                                                                   arrow=True,
                                                                   edge_color=edgeColorList)
            self.edgeCollections.append((self.network.edges(), edgeCollection))
            self.arrowCollections.append((self.network.edges(), arrowCollection))

        additionalNodeLabels = self.get_additional_node_labels()
        if additionalNodeLabels:
            # Plot Nash Flow Node Labels
            offset = (0, 8)
            add_tuple_offset = lambda a: (a[0] + offset[0], a[1] + offset[1])
            movedPositions = {node: add_tuple_offset(positions[node]) for node in positions}
            self.additionalNodeLabelCollection = draw_networkx_labels(self.network, pos=movedPositions, ax=self.axes,
                                                                      labels=additionalNodeLabels)
        else:
            self.additionalNodeLabelCollection = {}

        self.edgeLabels = self.get_edge_labels()

        self.edgeLabelCollection = draw_networkx_edge_labels(self.network, pos=positions, ax=self.axes,
                                                             edge_labels=self.edgeLabels, font_size=edgeLabelSize)

        self.draw_idle()

    def update_plot(self):
        self.update_nodes()
        self.update_edges()

    def update_nodes(self, added=False, removal=False, moved=False, color=False):
        nodeLabelSize = int(round(self.nodeLabelFontSize))
        if removal or moved:
            # A node has been deleted
            v = self.focusNode
            collectionIndex = 0
            for nodes, nodeCollection in self.nodeCollections:
                if v in nodes:
                    nodeCollection.remove()
                    nodes = [node for node in nodes if node != v]
                    if nodes:
                        positions = {v: self.network.node[v]['position'] for v in self.network.nodes()}
                        newNodeCollection = draw_networkx_nodes(self.network,
                                                                pos=positions,
                                                                ax=self.axes, node_size=self.nodeSize,
                                                                nodelist=nodes, node_color='r')
                        self.nodeCollections[collectionIndex] = (nodes, newNodeCollection)
                    else:
                        del self.nodeCollections[collectionIndex]

                    break

                collectionIndex += 1
            if not moved:
                # Delete node label
                deletedLabel = self.nodeLabelCollection.pop(self.focusNode)
                deletedLabel.remove()

            else:
                self.nodeCollections.append(([v], draw_networkx_nodes(self.network,
                                                                      pos={v: self.network.node[v]['position']},
                                                                      ax=self.axes, node_size=self.nodeSize,
                                                                      nodelist=[v], node_color='b')))
        elif added:
            # A node has been added (can we do better than plotting all nodes again)
            if self.focusNode is not None and all([self.focusNode not in entry for entry in self.nodeCollections]):
                v = self.focusNode
                self.nodeCollections.append(([v], draw_networkx_nodes(self.network,
                                                                      pos={v: self.network.node[v]['position']},
                                                                      ax=self.axes, node_size=self.nodeSize,
                                                                      nodelist=[v])))
                self.nodeLabelCollection.update(
                    draw_networkx_labels(self.network, pos={v: self.network.node[v]['position']}, ax=self.axes,
                                         labels={v: self.network.node[v]['label']}, font_size=nodeLabelSize))

        # Update node label texts and positions
        for v, label in self.nodeLabelCollection.iteritems():  # type(label) = matplotlib.text.Text object
            if label.get_text() != self.network.node[v]['label']:
                label.remove()
                self.nodeLabelCollection[v] = \
                    draw_networkx_labels(self.network, pos={v: self.network.node[v]['position']}, ax=self.axes,
                                         labels={v: self.network.node[v]['label']}, font_size=nodeLabelSize)[v]
            elif v == self.focusNode and moved:
                label.set_position(self.network.node[v]['position'])

        if color:
            nodeColor = lambda v: 'r' if v != self.focusNode else 'b'
            # Update colors and position
            for nodes, nodeCollection in self.nodeCollections:
                nodeColorList = [nodeColor(v) for v in nodes] if not removal else 'r'
                nodeCollection.set_facecolors(nodeColorList)

        self.draw_idle()

    def update_edges(self, added=False, removal=False, moved=False, color=False):
        if removal:
            # Edges have been deleted
            collectionIndex = 0
            toDeleteIndices = []
            for edges, edgeCollection in self.edgeCollections:
                missingEdges = [edge for edge in edges if edge not in self.network.edges()]
                if missingEdges:
                    arrowCollection = self.arrowCollections[collectionIndex][1]
                    edgeCollection.remove()
                    arrowCollection.remove()
                    edges = [edge for edge in edges if edge not in missingEdges]
                    if edges:
                        positions = {v: self.network.node[v]['position'] for v in self.network.nodes()}
                        newEdgeCollection, newArrowCollection = self.draw_edges(self.network, pos=positions,
                                                                                     ax=self.axes, arrow=True,
                                                                                     edgelist=edges)
                        self.edgeCollections[collectionIndex] = (edges, newEdgeCollection)
                        self.arrowCollections[collectionIndex] = (edges, newArrowCollection)

                    else:
                        toDeleteIndices.append(collectionIndex)

                    # Delete edge labels
                    for edge in missingEdges:
                        deletedLabel = self.edgeLabelCollection.pop(edge)
                        deletedLabel.remove()

                collectionIndex += 1

            for index in reversed(toDeleteIndices):
                del self.edgeCollections[index]
                del self.arrowCollections[index]


        elif added:
            # A node has been added (can we do better than plotting all nodes again)
            if self.focusEdge is not None:
                v, w = self.focusEdge
                edgeCollection, arrowCollection = self.draw_edges(self.network,
                                                                       pos={v: self.network.node[v]['position'],
                                                                            w: self.network.node[w]['position']},
                                                                       ax=self.axes, arrow=True,
                                                                       edgelist=[self.focusEdge])

                self.edgeCollections.append(([self.focusEdge], edgeCollection))
                self.arrowCollections.append(([self.focusEdge], arrowCollection))
                edgeLabelSize = int(round(self.edgeLabelFontSize))
                lbl = {self.focusEdge: (self.network[v][w]['capacity'], self.network[v][w]['transitTime'])}
                self.edgeLabelCollection.update(draw_networkx_edge_labels(self.network,
                                                                          pos={v: self.network.node[v]['position'],
                                                                               w: self.network.node[w]['position']},
                                                                          ax=self.axes, edge_labels=lbl,
                                                                          font_size=edgeLabelSize))

        elif moved:
            collectionIndex = 0
            for edges, edgeCollection in self.edgeCollections:
                pos = nx.get_node_attributes(self.network, 'position')
                edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edges])
                # Move edges
                edgeCollection.set_segments(edge_pos)

                a_pos = []
                p = 1.0 - 0.25  # make head segment 25 percent of edge length
                for src, dst in edge_pos:
                    x1, y1 = src
                    x2, y2 = dst
                    dx = x2 - x1  # x offset
                    dy = y2 - y1  # y offset
                    d = np.sqrt(float(dx ** 2 + dy ** 2))  # length of edge
                    if d == 0:  # source and target at same position
                        continue
                    if dx == 0:  # vertical edge
                        xa = x2
                        ya = dy * p + y1
                    if dy == 0:  # horizontal edge
                        ya = y2
                        xa = dx * p + x1
                    else:
                        theta = np.arctan2(dy, dx)
                        xa = p * d * np.cos(theta) + x1
                        ya = p * d * np.sin(theta) + y1

                    a_pos.append(((xa, ya), (x2, y2)))

                arrowCollection = self.arrowCollections[collectionIndex][1]
                # Move arrows
                arrowCollection.set_segments(a_pos)

                collectionIndex += 1

        if color:
            # Update colors
            edgeColor = lambda v, w: 'black' if (v, w) != self.focusEdge else 'b'
            collectionIndex = 0
            for edges, edgeCollection in self.edgeCollections:
                if edges:
                    edgeColorList = [colorConverter.to_rgba(edgeColor(v, w), 1) for v, w in
                                     edges] if not removal else 'black'
                    edgeCollection.set_color(edgeColorList)

                    arrowCollection = self.arrowCollections[collectionIndex][1]
                    arrowCollection.set_color(edgeColorList)
                    # edgeCollection.set_facecolors(edgeColorList)
                collectionIndex += 1

        # Update edge label texts and positions
        lbls = self.get_edge_labels()
        for edge, label in self.edgeLabelCollection.iteritems():  # type(label) = matplotlib.text.Text object
            v, w = edge
            lblTuple = lbls[(v, w)]  # (self.network[v][w]['capacity'], self.network[v][w]['transitTime'])
            #                lblTuple = "%.2f" % self.NTFEdgeFlowDict[edge]
            if label.get_text() != lblTuple:
                label.set_text(lblTuple)
            posv = (self.network.node[v]['position'][0] * 0.5, self.network.node[v]['position'][1] * 0.5)
            posw = (self.network.node[w]['position'][0] * 0.5, self.network.node[w]['position'][1] * 0.5)
            pos = (posv[0] + posw[0], posv[1] + posw[1])
            label.set_position(pos)

            # label.set_rotation(0.0)

        self.draw_idle()

    def zoom(self, factor):

        smaller = lambda val: factor * val  # Returns smaller value if factor < 1, i.e. if zooming out
        bigger = lambda val: (1. / factor) * val  # Returns bigger value if factor < 1, i.e. if zooming out

        # Scale axis
        self.Xlim = tuple(bigger(entry) for entry in self.Xlim)
        self.Ylim = tuple(bigger(entry) for entry in self.Ylim)
        self.axes.set_xlim(self.Xlim)
        self.axes.set_ylim(self.Ylim)

        # Scale node size
        self.nodeSize = smaller(self.nodeSize)
        for nodes, nodeCollection in self.nodeCollections:
            #nodeCollection.set_sizes([self.nodeSize for node in nodes])
            nodeCollection.set_sizes([self.nodeSize]*len(nodes))

        # Scale font size of node labels
        self.nodeLabelFontSize = smaller(self.nodeLabelFontSize)
        nodeLabelSize = int(round(self.nodeLabelFontSize))
        for v, label in self.nodeLabelCollection.iteritems():
            label.set_fontsize(nodeLabelSize)

        # Scale font size of edge labels
        self.edgeLabelFontSize = smaller(self.edgeLabelFontSize)
        edgeLabelSize = int(round(self.edgeLabelFontSize))
        for edge, label in self.edgeLabelCollection.iteritems():
            label.set_fontsize(edgeLabelSize)

        # Scale font size of Additional Node Labels, if existing
        for v, label in self.additionalNodeLabelCollection.iteritems():
            label.set_fontsize(nodeLabelSize)

        self.draw_idle()

    def move(self):
        dx = self.mouseWheelPosition[0] - self.mouseWheelPressedPosition[0]
        dy = self.mouseWheelPosition[1] - self.mouseWheelPressedPosition[1]

        self.Xlim = tuple(entry - dx for entry in self.Xlim)
        self.Ylim = tuple(entry - dy for entry in self.Ylim)

        self.axes.set_xlim(self.Xlim)
        self.axes.set_ylim(self.Ylim)

    def draw_edges(self, G, pos,
                   edgelist=None,
                   width=1.0,
                   edge_color='k',
                   style='solid',
                   alpha=1.0,
                   edge_cmap=None,
                   edge_vmin=None,
                   edge_vmax=None,
                   ax=None,
                   arrows=True,
                   label=None,
                   **kwds):

        return Utilities.draw_edges(G, pos,
                   edgelist,
                   width,
                   edge_color,
                   style,
                   alpha,
                   edge_cmap,
                   edge_vmin,
                   edge_vmax,
                   ax,
                   arrows,
                   label)

