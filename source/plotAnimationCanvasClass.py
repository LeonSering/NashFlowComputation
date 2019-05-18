# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         plotAnimationCanvasClass.py
# Description:  Class to extend plotCanvas in order to visualize animation
# ===========================================================================

from plotCanvasClass import PlotCanvas
from networkx import draw_networkx_labels, get_node_attributes
from utilitiesClass import Utilities
from bisect import insort
import os
from tempfile import gettempdir
import numpy as np
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import colorConverter
from matplotlib.animation import FFMpegFileWriter

# ======================================================================================================================

class PlotAnimationCanvas(PlotCanvas):
    """Class to extend plotCanvas in order to visualize animation"""
    def __init__(self, nashflow, interface, upperBound, stretchFactor):
        """
        :param nashflow: NashFlow instance
        :param interface: Interface instance
        :param upperBound: upper bound for the animation
        :param stretchFactor: factor to stretch canvas to avoid distortion
        """

        self.nashFlow = nashflow
        self.upperBound = upperBound
        self.network = self.nashFlow.network
        self.currentTimeIndex = 0
        self.maxTimeIndex = 99

        # Contains all times of animation
        self.timePoints = [(float(i) / self.maxTimeIndex) * self.upperBound for i in range(self.maxTimeIndex + 1)]

        self.NTFColors = ['seagreen', 'darkorange', 'aquamarine', 'deepskyblue', 'mediumpurple']

        # Dicts to save animation information
        self.nodeLabelByTimeDict = {node: dict() for node in self.network.nodes()}
        self.flowOnEntireEdge = {edge:{i:dict() for i in range(len(self.nashFlow.flowIntervals))} for edge in self.network.edges()}
        self.flowOnEdgeNotQueue = {edge: {i: dict() for i in range(len(self.nashFlow.flowIntervals))} for edge in
                                 self.network.edges()}
        self.flowOnQueue = {edge: {i: dict() for i in range(len(self.nashFlow.flowIntervals))} for edge in
                                 self.network.edges()}

        self.maxWidthFlowSize = {}
        self.maxOverflowStorage = {}
        for edge in self.network.edges():
            v, w = edge
            try:
                referenceCapacity = self.network[v][w]['inCapacity']
                referenceStorage = self.network[v][w]['storage']
            except KeyError:
                # General case
                referenceCapacity = float('inf')
                referenceStorage = float('inf')
            self.maxWidthFlowSize[edge] = referenceCapacity
            self.maxOverflowStorage[edge] = referenceStorage

        self.widthReferenceSize = {edge:{} for edge in self.network.edges()}

        #self.boxColoring = {edge:None for edge in self.network.edges()}
        self.edgeColoring = {edge:{} for edge in self.network.edges()}
        self.visualQueueColoring = {edge:{} for edge in self.network.edges()}
        self.precompute_information()

        PlotCanvas.__init__(self, graph=self.network, interface=interface, stretchFactor=stretchFactor)  # Call parents constructor

        positions = get_node_attributes(self.network, 'position')
        offset = (0, 8)
        add_tuple_offset = lambda a: (a[0] + offset[0], a[1] + offset[1])
        self.movedPositions = {node: add_tuple_offset(positions[node]) for node in positions}
        self.edgeWidthSize = 6
        self.tubeWidthFactor = 1
        self.tubeWidthMaximum = 0.6 * 14 # This is the size of the flow
        self.tubeWidthMinimum = 2
        self.zoom(factor=None)

    def precompute_information(self, timeList=None):
        """
        Compute information needed for the animation
        :param timeList: list of times for which information is computed. If None, then self.timePoints is used
        """
        self.maxQueueSize = 0
        self.maxInflowOnAllEdges = 0
        self.maxFlowOnEntireEdge = {edge:0 for edge in self.network.edges()}

        if not timeList:
            # Circumvent that default arguments are evaluated at function definition time
            timeList = self.timePoints

        for time in timeList:
            # Node Labels
            for v in self.network.nodes():
                self.nodeLabelByTimeDict[v][time] = self.nashFlow.node_label(v, time)

        for fk in range(len(self.nashFlow.flowIntervals)):
            # Compute information for each flowInterval fk
            for edge in self.network.edges():
                v, w = edge

                transitTime = self.network[v][w]['transitTime']
                inflowInterval, outflowInterval = self.nashFlow.animationIntervals[edge][fk]
                vTimeLower, vTimeUpper = inflowInterval
                wTimeLower, wTimeUpper = outflowInterval
                try:
                    inflow = self.network[v][w]['inflow'][inflowInterval]  # Could this lead to KeyError?
                    outflow = self.network[v][w]['outflow'][outflowInterval]
                    self.maxInflowOnAllEdges = max(inflow, self.maxInflowOnAllEdges)
                except:
                    raise KeyError("Label in corresponding NTF seems to be 0.")
                    inflow = 0
                    outflow = 0


                for time in timeList:
                    flowOnEntireEdge = max(0, max(0, inflow*(min(time, vTimeUpper)-vTimeLower))
                                                                 - max(0, outflow*(min(time, wTimeUpper)-wTimeLower)))
                    self.flowOnEntireEdge[edge][fk][time] = flowOnEntireEdge
                    self.maxFlowOnEntireEdge[edge] = max(flowOnEntireEdge, self.maxFlowOnEntireEdge[edge])

                    # Box at end
                    if not (vTimeLower <= time <= vTimeUpper + transitTime):
                        flowOnEdgeNotQueue = 0
                    elif time <= vTimeLower + transitTime:
                        flowOnEdgeNotQueue = inflow*(time-vTimeLower)
                    else:
                        # vTimeLower + transitTime < time < vTimeUpper + transitTime holds
                        flowOnEdgeNotQueue = inflow*(transitTime - max(0, time-vTimeUpper))

                    self.flowOnEdgeNotQueue[edge][fk][time] = flowOnEdgeNotQueue

                    flowOnQueue = max(0, flowOnEntireEdge-flowOnEdgeNotQueue)   # Box at end

                    self.flowOnQueue[edge][fk][time] = flowOnQueue

        for time in timeList:
            for edge in self.network.edges():
                v, w = edge
                m = 0
                for fk in range(len(self.nashFlow.flowIntervals)):
                    m += self.flowOnQueue[edge][fk][time]

                self.maxQueueSize = max(self.maxQueueSize, m)

        for edge in self.network.edges():
            if self.maxWidthFlowSize[edge] == float('inf'):
                self.maxWidthFlowSize[edge] = self.maxInflowOnAllEdges + 1
            if self.maxOverflowStorage[edge] == float('inf'):
                self.maxOverflowStorage[edge] == self.maxFlowOnEntireEdge[edge] + 1

    def reset_bounds(self, lowerBound, upperBound):
        """
        Resets animation time bounds
        :param lowerBound: lower time bound of animation
        :param upperBound: upper time bound of animation
        """
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.timePoints = [self.lowerBound + float(i) / self.maxTimeIndex * (self.upperBound-self.lowerBound) for i in range(self.maxTimeIndex + 1)]
        self.precompute_information()
        self.update_time_labels()

    def add_time(self, time):
        """
        Add time to self.timePoints
        """
        insort(self.timePoints, time)
        self.maxTimeIndex += 1
        self.precompute_information(timeList=[time])

    def time_changed(self, sliderVal):
        """
        Update the animation given a time index
        :param sliderVal: index
        """
        self.currentTimeIndex = sliderVal
        self.update_time_labels()
        self.update_flow_animation()

    def get_time_from_tick(self, sliderVal):
        """
        :param sliderVal: time index
        :return: time corresponding to time index sliderVal
        """
        return self.timePoints[sliderVal]

    def get_flowinterval_index_from_tick(self, sliderVal):
        """
        Get index of flowInterval in list self.nashFlow.flowIntervals corresponding to time index sliderVal
        :param sliderVal: time index
        :return: -1 if not found, otherwise the flowInterval in which self.get_time_from_tick(sliderVal) lies
        """
        t = self.get_time_from_tick(sliderVal)
        for index, interval in enumerate(self.nashFlow.flowIntervals):
            if interval[0] <= t <= interval[1]:
                return index
        return -1

    def update_time_labels(self):
        """Update additional node labels"""
        nodeLabelSize = int(round(self.nodeLabelFontSize))

        for v, label in self.additionalNodeLabelCollection.iteritems():  # type(label) = matplotlib.text.Text object
            label.remove()

        self.additionalNodeLabelCollection = draw_networkx_labels(self.network, pos=self.movedPositions, ax=self.axes,
                                                                  labels=self.get_additional_node_labels(),
                                                                  font_size=nodeLabelSize)
        self.draw_idle()

    def update_flow_animation(self):
        """Update the animation"""
        for edge in self.network.edges():
            v,w = edge
            src, dst = self.network.node[v]['position'], self.network.node[w]['position']
            #self.draw_queue_color_box(edge, src, dst)
            #self.draw_edge_colors(edge, src, dst)
            self.draw_flow(edge, src, dst)
        self.draw_idle()

    def draw_flow(self, edge, src, dst):
        """
        Draw flow animation
        :param edge: edge = vw to draw
        :param src: position of v
        :param dst: position of w
        """
        if self.edgeColoring[edge]:
            for fk in self.edgeColoring[edge].keys():
                self.edgeColoring[edge][fk].remove()
        self.edgeColoring[edge].clear()
        if self.widthReferenceSize[edge]:
            self.widthReferenceSize[edge].clear()
        if self.visualQueueColoring[edge]:
            self.visualQueueColoring.remove()

        v, w = edge
        time = self.timePoints[self.currentTimeIndex]
        transitTime = self.network[v][w]['transitTime']
        src = np.array(src)
        dst = np.array(dst)
        s = dst - src

        flowOnEdgeList = [self.flowOnEntireEdge[edge][fk][time] for fk in range(len(self.nashFlow.flowIntervals))]
        totalFlowOnEdge = sum(flowOnEdgeList)
        if Utilities.is_eq_tol(totalFlowOnEdge, 0):
            return

        overflowBlocks = []
        for fk in range(len(self.nashFlow.flowIntervals)):
            if Utilities.is_eq_tol(self.flowOnEdgeNotQueue[edge][fk][time], 0):
                # No flow on edgeNotQueue at this time
                continue

            inflowInterval, outflowInterval = self.nashFlow.animationIntervals[edge][fk]
            vTimeLower, vTimeUpper = inflowInterval
            inflow = float(self.network[v][w]['inflow'][(vTimeLower, vTimeUpper)])

            # These position factors are using that the amount on the edgeNotQueue has to be positive at this point
            # This implies that vTimeLower <= time <= vTimeUpper + transitTime

            startFac = float(time - vTimeUpper) / transitTime if time > vTimeUpper else 0
            endFac = float(time - vTimeLower) / transitTime if time <= vTimeLower + transitTime else 1

            start = src + startFac * s
            end = src + endFac * s

            edge_pos = np.asarray([(start, end)])
            edge_color = tuple(colorConverter.to_rgba(self.NTFColors[fk % len(self.NTFColors)],
                                                      alpha=1))

            widthRatioScale = min(1, inflow/self.maxWidthFlowSize[edge])
            self.widthReferenceSize[edge][fk] = max(self.tubeWidthMaximum*widthRatioScale, self.tubeWidthMinimum)
            # Drawing of flow tubes
            edgeCollection = LineCollection(edge_pos,
                                            colors=edge_color,
                                            linewidths=self.tubeWidthFactor*self.widthReferenceSize[edge][fk],
                                            antialiaseds=(1,),
                                            transOffset=self.axes.transData,
                                            alpha=1
                                            )

            edgeCollection.set_zorder(1)
            self.edgeColoring[edge][fk] = edgeCollection
            self.axes.add_collection(edgeCollection)

            # Drawing of overflow blocks
            '''
            if not overflowBlocks:
                if Utilities.is_eq_tol(self.flowOnQueue[edge][fk][time], 0):
                    continue
                else:
                    # Draw first block
                    blockSizeFactor = min(1, self.flowOnEntireEdge[edge][fk][time]/self.maxOverflowStorage[edge])

                    block = Rectangle(start - delta,
                                    width=d,
                                    height=14,
                                    transform=t,
                                    facecolor=self.NTFColors[fk % len(self.NTFColors)],
                                    linewidth=None,
                                    alpha=1)
                    overflowBlocks.append(block)
            else:
                pass
            '''

        if overflowBlocks:
            overflowBlockCollection = PatchCollection(boxes,
                                            match_original=True,
                                            antialiaseds=(1,),
                                            transOffset=self.axes.transData)
            self.visualQueueColoring[edge] = overflowBlockCollection
            self.axes.add_collection(overflowBlockCollection)

    def get_additional_node_labels(self):
        """Return node label dict"""
        return {node: "%.2f" % self.nodeLabelByTimeDict[node][self.timePoints[self.currentTimeIndex]] for node in
                self.network.nodes()}

    def get_edge_labels(self):
        """Return edge label dict"""
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
            clickedNode = self.check_node_clicked((xAbsolute, yAbsolute), edgePossible=True)  # Never add a new node

            if clickedEdge is not None and clickedNode is None:
                # Selected an existing edge
                self.focusEdge = clickedEdge
                self.focusNode = None
                self.interface.update_edge_diagrams()
                self.interface.update_node_label_diagram()
                self.interface.update_plotanimationcanvas_focusselection()
                self.update_edges(color=True)
                self.update_nodes(color=True)
            elif clickedNode is not None:
                self.focusNode = clickedNode
                self.focusEdge = None
                self.mouseLeftPressTime = None
                self.update_nodes(added=False, color=True)
                self.update_edges(color=True)
                self.interface.update_plotanimationcanvas_focusselection()
                self.interface.update_node_label_diagram()
                self.interface.update_edge_diagrams()

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
        """Workaround to call specific edge drawing function"""

        edges, boxes, tubes = Utilities.draw_animation_edges(G, pos,
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

        self.tube_collection = tubes
        return edges, boxes

    def zoom(self, factor=None):
        """Zoom by a factor"""
        if factor is not None:
            smaller = lambda val: factor * val  # Returns smaller value if factor < 1, i.e. if zooming out
            bigger = lambda val: (1. / factor) * val  # Returns bigger value if factor < 1, i.e. if zooming out
        else:
            smaller = lambda val: val
            bigger = lambda val: val

        # Scale axis
        self.Xlim = tuple(bigger(entry) for entry in self.Xlim)
        self.Ylim = tuple(bigger(entry) for entry in self.Ylim)
        self.axes.set_xlim(self.Xlim)
        self.axes.set_ylim(self.Ylim)

        # Scale edge widths
        self.edgeWidthSize = smaller(self.edgeWidthSize)
        #edgeWithSize = max(1, int(round(self.edgeWidthSize)))
        for edges, edgeCollection in self.edgeCollections:
            edgeCollection.set_linewidth(self.edgeWidthSize)

        # Scale tubes
        self.tubeWidthFactor = smaller(self.tubeWidthFactor)
        #self.tube_collection.set_linewidth(self.tubeWidthSize)

        # Scale colored edges if existing
        for edge in self.network.edges():
            edgeCollection = self.edgeColoring[edge]
            for fk, fkEdgeColoring in edgeCollection.items():
                fkEdgeColoring.set_linewidth(self.tubeWidthFactor*self.widthReferenceSize[edge][fk])

        # Scale font size of node labels
        self.nodeLabelFontSize = smaller(self.nodeLabelFontSize)
        nodeLabelSize = int(round(self.nodeLabelFontSize))
        for v, label in self.nodeLabelCollection.iteritems():
            label.set_fontsize(nodeLabelSize)

        # Scale font size of Additional Node Labels, if existing
        for v, label in self.additionalNodeLabelCollection.iteritems():
            label.set_fontsize(nodeLabelSize)

        # Scale font size of edge labels
        self.edgeLabelFontSize = smaller(self.edgeLabelFontSize)
        edgeLabelSize = int(round(self.edgeLabelFontSize))
        for edge, label in self.edgeLabelCollection.iteritems():
            label.set_fontsize(edgeLabelSize)

        self.draw_idle()

    def export(self, path):
        """Export mp4-video of animation to path. This requires FFMPEG."""
        currentTimeIndex = self.currentTimeIndex  # Index to jump back to
        ffmpegWriter = FFMpegFileWriter()
        with ffmpegWriter.saving(self.figure, path, dpi=100):
            for t_i in range(self.maxTimeIndex):
                self.time_changed(t_i)
                ffmpegWriter.grab_frame()
        self.time_changed(currentTimeIndex)

