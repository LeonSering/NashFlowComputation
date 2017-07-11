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
from bisect import insort
import numpy as np
import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import colorConverter


# ======================================================================================================================



class PlotAnimationCanvas(PlotCanvas):
    def __init__(self, nashflow, interface, upperBound, stretchFactor):
        self.nashFlow = nashflow
        self.upperBound = upperBound
        self.network = self.nashFlow.network
        self.currentTimeIndex = 0
        self.maxTimeIndex = 99
        self.timePoints = [float(i) / self.maxTimeIndex * self.upperBound for i in range(self.maxTimeIndex + 1)]
        self.nodeLabelByTimeDict = {node: dict() for node in self.network.nodes()}
        self.NTFColors = ['r', 'b', 'g', 'orange']


        self.flowOnEntireEdge = {edge:{i:dict() for i in range(len(self.nashFlow.flowIntervals))} for edge in self.network.edges()}
        self.flowOnEdgeNotQueue = {edge: {i: dict() for i in range(len(self.nashFlow.flowIntervals))} for edge in
                                 self.network.edges()}
        self.flowOnQueue = {edge: {i: dict() for i in range(len(self.nashFlow.flowIntervals))} for edge in
                                 self.network.edges()}

        self.boxColoring = {edge:None for edge in self.network.edges()}
        self.edgeColoring = {edge:None for edge in self.network.edges()}
        self.precompute_information()

        PlotCanvas.__init__(self, graph=self.network, interface=interface, stretchFactor=stretchFactor)  # Call parents constructor

        positions = get_node_attributes(self.network, 'position')
        offset = (0, 8)
        add_tuple_offset = lambda a: (a[0] + offset[0], a[1] + offset[1])
        self.movedPositions = {node: add_tuple_offset(positions[node]) for node in positions}

    def precompute_information(self, timeList=None):
        if not timeList:
            # Circumvent that default arguments are evaluated at function definition time
            timeList = self.timePoints

        for time in timeList:
            # Node Labels
            for v in self.network.nodes():
                self.nodeLabelByTimeDict[v][time] = self.nashFlow.node_label(v, time)

        for fk in range(len(self.nashFlow.flowIntervals)):
            for edge in self.network.edges():
                v, w = edge

                transitTime = self.network[v][w]['transitTime']
                inflowInterval, outflowInterval = self.nashFlow.animationIntervals[edge][fk]
                vTimeLower, vTimeUpper = inflowInterval
                wTimeLower, wTimeUpper = outflowInterval
                try:
                    inflow = self.network[v][w]['inflow'][inflowInterval]  # Could this lead to KeyError?
                    outflow = self.network[v][w]['outflow'][outflowInterval]
                except:
                    raise KeyError("Label in corresponding NTF seems to be 0.")
                    inflow = 0
                    outflow = 0


                for time in timeList:
                    flowOnEntireEdge = max(0, max(0, inflow*(min(time, vTimeUpper)-vTimeLower))
                                                                 - max(0, outflow*(min(time, wTimeUpper)-wTimeLower)))
                    self.flowOnEntireEdge[edge][fk][time] = flowOnEntireEdge

                    #self.flowOnEdgeNotQueue[edge][fk][time] = max(0, outflow*(min(time, wTimeUpper - transitTime)-(wTimeLower - transitTime)))
                    if wTimeLower - transitTime >= time or time >= wTimeUpper:
                        flowOnEdgeNotQueue = 0
                    elif wTimeLower - transitTime <= time <= wTimeLower:
                        flowOnEdgeNotQueue = outflow*(time - wTimeLower + transitTime)
                    elif wTimeLower <= time <= wTimeUpper - transitTime:
                        flowOnEdgeNotQueue = outflow*transitTime
                    elif wTimeUpper - transitTime <= time <= wTimeUpper:
                        flowOnEdgeNotQueue = outflow*(wTimeUpper-time)
                    self.flowOnEdgeNotQueue[edge][fk][time] = flowOnEdgeNotQueue

                    flowOnQueue = max(0, self.flowOnEntireEdge[edge][fk][time] - self.flowOnEdgeNotQueue[edge][fk][time])

                    self.flowOnQueue[edge][fk][time] = flowOnQueue


    def reset_bounds(self, lowerBound, upperBound):
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.timePoints = [self.lowerBound + float(i) / self.maxTimeIndex * (self.upperBound-self.lowerBound) for i in range(self.maxTimeIndex + 1)]
        self.precompute_information()
        self.update_time_labels()

    def add_time(self, time):
        insort(self.timePoints, time)
        self.maxTimeIndex += 1
        self.precompute_information(timeList=[time])

    def time_changed(self, sliderVal):
        self.currentTimeIndex = sliderVal
        self.update_time_labels()
        self.update_flow_animation()

    def get_time_from_tick(self, sliderVal):
        return self.timePoints[sliderVal]

    def update_time_labels(self):
        # Update additional node labels

        nodeLabelSize = int(round(self.nodeLabelFontSize))

        for v, label in self.additionalNodeLabelCollection.iteritems():  # type(label) = matplotlib.text.Text object
            label.remove()

        self.additionalNodeLabelCollection = draw_networkx_labels(self.network, pos=self.movedPositions, ax=self.axes,
                                                                  labels=self.get_additional_node_labels(),
                                                                  font_size=nodeLabelSize)
        self.draw_idle()

    def update_flow_animation(self):

        for edge in self.network.edges():
            v,w = edge
            src, dst = self.network.node[v]['position'], self.network.node[w]['position']
            self.draw_queue_color_box(edge, src, dst)
            self.draw_edge_colors(edge, src, dst)


        self.draw_idle()

    def draw_edge_colors(self, edge, src, dst, p=0.25):
        if self.edgeColoring[edge] is not None:
            self.edgeColoring[edge].remove()
        self.edgeColoring[edge] = None
        v, w = edge
        time = self.timePoints[self.currentTimeIndex]
        transitTime, capacity = self.network[v][w]['transitTime'], self.network[v][w]['capacity']
        maximumFlow = transitTime*capacity

        if maximumFlow == 0:
            return

        flowRatio = [
            max(0, float(self.flowOnEdgeNotQueue[edge][fk][self.timePoints[self.currentTimeIndex]]) / maximumFlow)
            for fk in range(len(self.nashFlow.flowIntervals))]

        src = np.array(src)
        dst = np.array(dst)
        srcAfterBox = src + p*(dst - src)
        s = dst-srcAfterBox
        edge_pos = []
        edge_colors = []
        for fk in range(len(self.nashFlow.flowIntervals)):
            if Utilities.is_eq_tol(flowRatio[fk], 0):
                continue
            inflowInterval, outflowInterval = self.nashFlow.animationIntervals[edge][fk]
            wTimeLower, wTimeUpper = outflowInterval

            # these position factors are using that the amount on the edgeNotQueue has to be positive at this point
            startFac = float(transitTime - (wTimeUpper-time))/transitTime if time >= wTimeUpper - transitTime else 0
            endFac = float(transitTime-(wTimeLower-time))/transitTime if wTimeLower - transitTime <= time <= wTimeLower else 1
            start = srcAfterBox + startFac * s
            end = srcAfterBox + endFac * s
            edge_pos.append((start, end))
            edge_colors.append(self.NTFColors[fk % len(self.NTFColors)])

        edge_pos = np.asarray(edge_pos)

        edge_colors = tuple([colorConverter.to_rgba(c, alpha=1)
                             for c in edge_colors])

        edgeCollection = LineCollection(edge_pos,
                                         colors=edge_colors,
                                         linewidths=self.edgeWidthSize*0.7,
                                         antialiaseds=(1,),
                                         transOffset=self.axes.transData,
                                         alpha = 1
                                         )

        edgeCollection.set_zorder(1)
        self.edgeColoring[edge] = edgeCollection
        self.axes.add_collection(edgeCollection)


    def draw_queue_color_box(self, edge, src, dst, p=0.25, radius=7, lastProportion=1):
        if self.boxColoring[edge] is not None:
            self.boxColoring[edge].remove()
        self.boxColoring[edge] = None

        totalFlowOnQueue = sum(self.flowOnQueue[edge][fk][self.timePoints[self.currentTimeIndex]] for fk in range(len(self.nashFlow.flowIntervals)))

        if Utilities.is_eq_tol(totalFlowOnQueue, 0):
            return

        flowRatio = [max(0, float(self.flowOnQueue[edge][fk][self.timePoints[self.currentTimeIndex]])/totalFlowOnQueue)
                      for fk in range(len(self.nashFlow.flowIntervals))]

        delta = np.array([0, radius])
        src = np.array(src)
        dst = np.array(dst)
        s = dst - src
        angle = np.rad2deg(np.arctan2(s[1], s[0]))
        t = matplotlib.transforms.Affine2D().rotate_deg_around(src[0], src[1], angle)
        boxes = []
        for fk in range(len(self.nashFlow.flowIntervals)):
            if Utilities.is_eq_tol(flowRatio[fk], 0):
                continue

            d = np.sqrt(np.sum(((dst - src) * p * lastProportion) ** 2))
            rec = Rectangle(src - delta,
                            width=d,
                            height=radius * 2,
                            transform=t,
                            facecolor=self.NTFColors[fk % len(self.NTFColors)],
                            linewidth=int(lastProportion),
                            alpha=1)
            boxes.append(rec)

            lastProportion -= flowRatio[fk]

        boxCollection = PatchCollection(boxes,
                                        match_original=True,
                                        antialiaseds=(1,),
                                        transOffset=self.axes.transData)
        self.boxColoring[edge] = boxCollection
        self.axes.add_collection(boxCollection)


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
                self.interface.update_plotAnimationCanvas_focusSelection()
                self.update_edges(color=True)
                self.update_nodes(color=True)
            elif clickedNode is not None:
                self.focusNode = clickedNode
                self.focusEdge = None
                self.mouseLeftPressTime = None
                self.update_nodes(added=True, color=True)
                self.update_edges(color=True)
                self.interface.update_plotAnimationCanvas_focusSelection()
                self.interface.update_node_label_graph()
                self.interface.update_edge_graphs()

        elif action == 2:
            # Wheel was clicked, move visible part of canvas
            self.currentXlim = self.Xlim
            self.currentYlim = self.Ylim
            self.mouseWheelPressed = True
            self.mouseWheelPressedPosition = (xAbsolute, yAbsolute)
            return

    def on_release(self, event):
        """
        Release-Mouse-event handling
        :param event: event which is emitted by matplotlib
        """
        if event.xdata is None or event.ydata is None:
            return
        xAbsolute, yAbsolute = event.xdata, event.ydata
        action = event.button  # event.button = mouse(1,2,3)


        if action == 1:
            # Nothing should happen here, as no new node can be created
            return

        elif action == 2:
            # Wheel has been released
            self.mouseWheelPressed = False
            self.mouseWheelPressedPosition = None

        elif action == 3:
            # Right mouse has been released
            self.mouseRightPressed = False
            self.selectedNode = None

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