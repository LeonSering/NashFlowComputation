# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotAnimationCanvasClass.py
# Description:
# Parameters:
# ===========================================================================

from plotCanvasClass import PlotCanvas
from networkx import draw_networkx_labels, get_node_attributes
# ======================================================================================================================



class PlotAnimationCanvas(PlotCanvas):

    def __init__(self, nashflow, interface, upperBound):
        self.nashFlow = nashflow
        self.upperBound = upperBound
        self.network = self.nashFlow.network
        self.currentTimeIndex = 0
        self.precompute_information()
        PlotCanvas.__init__(self, graph=self.network, interface=interface) # Call parents constructor

        positions = get_node_attributes(self.network, 'position')
        offset = (0, 8)
        add_tuple_offset = lambda a: (a[0] + offset[0], a[1] + offset[1])
        self.movedPositions = {node: add_tuple_offset(positions[node]) for node in positions}

    def precompute_information(self):
        self.timePoints = [float(i) / 99 * self.upperBound for i in range(100)]
        self.nodeLabelByTimeDict = {node: dict() for node in self.network.nodes()}

        # Node Labels
        for timeIndex in range(100):
            time = self.timePoints[timeIndex]
            for v in self.network.nodes():
                self.nodeLabelByTimeDict[v][time] = self.nashFlow.node_label(v, time)

    def time_changed(self, sliderVal):
        self.currentTimeIndex = sliderVal
        self.update_time_labels()

    def update_time_labels(self):
        # Update additional node labels and positions

        nodeLabelSize = int(round(self.nodeLabelFontSize))

        for v, label in self.additionalNodeLabelCollection.iteritems():    # type(label) = matplotlib.text.Text object
                label.remove()

        self.additionalNodeLabelCollection=draw_networkx_labels(self.network, pos=self.movedPositions, ax=self.axes,
                                 labels=self.get_additional_node_labels(), font_size=nodeLabelSize)

        self.draw_idle()


    def get_additional_node_labels(self):
        return {node: "%.2f" % self.nodeLabelByTimeDict[node][self.timePoints[self.currentTimeIndex]] for node in self.network.nodes()}

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

        action = event.button   # event.button = mouse(1,2,3)

        if action == 1:
            # Leftmouse was clicked, select/create node, select edge or add edge via drag&drop

            # Determine whether we clicked an edge or not
            clickedEdge = self.check_edge_clicked((xAbsolute, yAbsolute))


            # Determine whether we clicked a node or not
            clickedNode = self.check_node_clicked((xAbsolute, yAbsolute), edgePossible=True) # never add a new node

            if clickedEdge is not None and clickedNode is None:
                # Selected an existing edge
                self.focusEdge = clickedEdge
                self.interface.update_edge_graphs()
                self.update_edges(color=True)
            elif clickedNode is not None:
                self.focusNode = clickedNode
                self.mouseLeftPressTime = None
                self.update_nodes(added=True, color=True)
                self.interface.update_node_label_graph()

        elif action == 2:
            # Wheel was clicked, move visible part of canvas
            self.currentXlim = self.Xlim
            self.currentYlim = self.Ylim
            self.mouseWheelPressed = True
            self.mouseWheelPressedPosition = (xAbsolute, yAbsolute)
            return
