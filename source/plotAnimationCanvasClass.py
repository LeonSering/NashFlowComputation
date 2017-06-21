# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotAnimationCanvasClass.py
# Description:
# Parameters:
# ===========================================================================

from plotCanvasClass import PlotCanvas
from networkx import draw_networkx_labels, get_node_attributes
from utilitiesClass import Utilities


# ======================================================================================================================



class PlotAnimationCanvas(PlotCanvas):
    def __init__(self, nashflow, interface, upperBound):
        self.nashFlow = nashflow
        self.upperBound = upperBound
        self.network = self.nashFlow.network
        self.currentTimeIndex = 0
        self.precompute_information()

        PlotCanvas.__init__(self, graph=self.network, interface=interface)  # Call parents constructor

        positions = get_node_attributes(self.network, 'position')
        offset = (0, 8)
        add_tuple_offset = lambda a: (a[0] + offset[0], a[1] + offset[1])
        self.movedPositions = {node: add_tuple_offset(positions[node]) for node in positions}

    def precompute_information(self):
        self.timePoints = [float(i) / 99 * self.upperBound for i in range(100)]
        self.nodeLabelByTimeDict = {node: dict() for node in self.network.nodes()}


        self.flowOnEntireEdge = {edge:{i:dict() for i in range(len(self.nashFlow.flowIntervals))} for edge in self.network.edges()}
        self.flowOnEdgeNotQueue = {edge: {i: dict() for i in range(len(self.nashFlow.flowIntervals))} for edge in
                                 self.network.edges()}
        self.flowOnQueue = {edge: {i: dict() for i in range(len(self.nashFlow.flowIntervals))} for edge in
                                 self.network.edges()}

        for time in self.timePoints:
            # Node Labels
            for v in self.network.nodes():
                self.nodeLabelByTimeDict[v][time] = self.nashFlow.node_label(v, time)

        for fk in range(len(self.nashFlow.flowIntervals)):
            for edge in self.network.edges():
                for time in self.timePoints:
                    v, w = edge
                    inflowInterval, outflowInterval = self.nashFlow.animationIntervals[edge][fk]
                    vTimeLower, vTimeUpper = inflowInterval
                    wTimeLower, wTimeUpper = outflowInterval
                    capacity = self.network[v][w]['capacity']
                    inflow = self.network[v][w]['inflow'][inflowInterval] # Could this lead to KeyError?
                    outflow = self.network[v][w]['outflow'][outflowInterval]
                    self.flowOnEntireEdge[edge][fk][time] = max(0, max(0, inflow*(min(time, vTimeUpper)-vTimeLower))
                                                                 - max(0, outflow*(min(time, wTimeUpper)-wTimeLower)))
                    self.flowOnEdgeNotQueue[edge][fk][time] = max(0, outflow*(min(time, wTimeUpper - capacity)-(wTimeLower - capacity)))
                    self.flowOnQueue[edge][fk][time] = self.flowOnEntireEdge[edge][fk][time] - self.flowOnEdgeNotQueue[edge][fk][time]

    def time_changed(self, sliderVal):
        self.currentTimeIndex = sliderVal
        self.update_time_labels()

    def get_time_from_tick(self, sliderVal):
        return self.timePoints[sliderVal]

    def update_time_labels(self):
        # Update additional node labels and positions

        nodeLabelSize = int(round(self.nodeLabelFontSize))

        for v, label in self.additionalNodeLabelCollection.iteritems():  # type(label) = matplotlib.text.Text object
            label.remove()

        self.additionalNodeLabelCollection = draw_networkx_labels(self.network, pos=self.movedPositions, ax=self.axes,
                                                                  labels=self.get_additional_node_labels(),
                                                                  font_size=nodeLabelSize)

        self.draw_idle()

    def get_additional_node_labels(self):
        return {node: "%.2f" % self.nodeLabelByTimeDict[node][self.timePoints[self.currentTimeIndex]] for node in
                self.network.nodes()}

    def get_edge_labels(self):
        return {}

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

            # Determine whether we clicked an edge or not
            clickedEdge = self.check_edge_clicked((xAbsolute, yAbsolute))

            # Determine whether we clicked a node or not
            clickedNode = self.check_node_clicked((xAbsolute, yAbsolute), edgePossible=True)  # never add a new node

            if clickedEdge is not None and clickedNode is None:
                # Selected an existing edge
                self.focusEdge = clickedEdge
                self.focusNode = None
                self.interface.update_edge_graphs()
                self.interface.update_node_label_graph()
                self.update_edges(color=True)
                self.update_nodes(color=True)
            elif clickedNode is not None:
                self.focusNode = clickedNode
                self.focusEdge = None
                self.mouseLeftPressTime = None
                self.update_nodes(added=True, color=True)
                self.update_edges(color=True)
                self.interface.update_node_label_graph()
                self.interface.update_edge_graphs()

        elif action == 2:
            # Wheel was clicked, move visible part of canvas
            self.currentXlim = self.Xlim
            self.currentYlim = self.Ylim
            self.mouseWheelPressed = True
            self.mouseWheelPressedPosition = (xAbsolute, yAbsolute)
            return

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

        return Utilities.draw_animation_edges(G, pos,
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


    def get_inflow_interval(self, edge, time):
        v,w = edge
        for lowerBound, upperBound in self.network[v][w]['inflow']:
            if lowerBound <= time <= upperBound:
                return (lowerBound, upperBound)

        return -1


    def get_outflow_interval(self, edge, time):
        v,w = edge
        for lowerBound, upperBound in self.network[v][w]['outflow']:
            if lowerBound <= time <= upperBound:
                return (lowerBound, upperBound)

        return -1