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
import numpy as np
import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.colors import colorConverter

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

        self.maximalQueueSize = 0

        self.boxColoring = {edge:None for edge in self.network.edges()}
        self.edgeColoring = {edge:None for edge in self.network.edges()}
        self.precompute_information()

        PlotCanvas.__init__(self, graph=self.network, interface=interface, stretchFactor=stretchFactor)  # Call parents constructor

        positions = get_node_attributes(self.network, 'position')
        offset = (0, 8)
        add_tuple_offset = lambda a: (a[0] + offset[0], a[1] + offset[1])
        self.movedPositions = {node: add_tuple_offset(positions[node]) for node in positions}
        self.edgeWidthSize = 6
        self.tubeWidthSize = 4
        self.zoom(factor=None)

    def precompute_information(self, timeList=None):
        """
        Compute information needed for the animation
        :param timeList: list of times for which information is computed. If None, then self.timePoints is used
        """
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
                except:
                    raise KeyError("Label in corresponding NTF seems to be 0.")
                    inflow = 0
                    outflow = 0


                for time in timeList:
                    flowOnEntireEdge = max(0, max(0, inflow*(min(time, vTimeUpper)-vTimeLower))
                                                                 - max(0, outflow*(min(time, wTimeUpper)-wTimeLower)))
                    self.flowOnEntireEdge[edge][fk][time] = flowOnEntireEdge
                    '''
                    # Box at beginning
                    if wTimeLower - transitTime >= time or time >= wTimeUpper:
                        flowOnEdgeNotQueue = 0
                    elif wTimeLower - transitTime <= time <= wTimeLower:
                        flowOnEdgeNotQueue = outflow*(time - wTimeLower + transitTime)
                    elif wTimeLower <= time <= wTimeUpper - transitTime:
                        flowOnEdgeNotQueue = outflow*transitTime
                    elif wTimeUpper - transitTime <= time <= wTimeUpper:
                        flowOnEdgeNotQueue = outflow*(wTimeUpper-time)
                    '''
                    # Box at end
                    if not (vTimeLower <= time <= vTimeUpper + transitTime):
                        flowOnEdgeNotQueue = 0
                    elif time <= vTimeLower + transitTime:
                        flowOnEdgeNotQueue = inflow*(time-vTimeLower)
                    else:
                        # vTimeLower + transitTime < time < vTimeUpper + transitTime holds
                        flowOnEdgeNotQueue = inflow*transitTime

                    self.flowOnEdgeNotQueue[edge][fk][time] = flowOnEdgeNotQueue

                    #flowOnQueue = max(0, self.flowOnEntireEdge[edge][fk][time] - self.flowOnEdgeNotQueue[edge][fk][time])  # Box at beginning
                    flowOnQueue = max(0, flowOnEntireEdge-flowOnEdgeNotQueue)   # Box at end

                    self.flowOnQueue[edge][fk][time] = flowOnQueue

        for time in timeList:
            for edge in self.network.edges():
                v, w = edge
                m = 0
                for fk in range(len(self.nashFlow.flowIntervals)):
                    m += self.flowOnQueue[edge][fk][time]

                self.maximalQueueSize = max(self.maximalQueueSize, m)

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
            self.draw_edge_colors(edge, src, dst)
        self.draw_idle()

    def draw_edge_colors(self, edge, src, dst, p=0.25):
        """
        Draw colors of flow animation
        :param edge: edge = vw to draw
        :param src: position of v
        :param dst: position of w
        :param p: length of box
        """
        if self.edgeColoring[edge] is not None:
            self.edgeColoring[edge].remove()
        self.edgeColoring[edge] = None
        v, w = edge
        time = self.timePoints[self.currentTimeIndex]
        transitTime, capacity = self.network[v][w]['transitTime'], self.network[v][w]['outCapacity']
        src = np.array(src)
        dst = np.array(dst)
        '''
        # Box at beginning
        maximumFlow = transitTime*capacity  # Box at beginning

        if maximumFlow == 0:
            return

        flowRatio = [
            max(0, float(self.flowOnEdgeNotQueue[edge][fk][time]) / maximumFlow)
            for fk in range(len(self.nashFlow.flowIntervals))]
        
        srcAfterBox = src + p*(dst - src) # Box at beginning
        '''

        dstBeforeBox = src + (1-p)*(dst - src)    # Box at the end
        s = dstBeforeBox-src
        edge_pos = []
        edge_colors = []

        for fk in range(len(self.nashFlow.flowIntervals)):
            '''
            # Box at beginning
            if Utilities.is_eq_tol(flowRatio[fk], 0):
                continue
            inflowInterval, outflowInterval = self.nashFlow.animationIntervals[edge][fk]
            wTimeLower, wTimeUpper = outflowInterval

            # These position factors are using that the amount on the edgeNotQueue has to be positive at this point
            startFac = float(transitTime - (wTimeUpper-time))/transitTime if time >= wTimeUpper - transitTime else 0
            endFac = float(transitTime-(wTimeLower-time))/transitTime if wTimeLower - transitTime <= time <= wTimeLower else 1
            start = srcAfterBox + startFac * s
            end = srcAfterBox + endFac * s
            edge_pos.append((start, end))
            edge_colors.append(self.NTFColors[fk % len(self.NTFColors)])
            '''

            # Box at end
            if Utilities.is_eq_tol(self.flowOnEdgeNotQueue[edge][fk][time], 0):
                # No flow on edgeNotQueue at this time
                continue
            inflowInterval, outflowInterval = self.nashFlow.animationIntervals[edge][fk]
            vTimeLower, vTimeUpper = inflowInterval

            # These position factors are using that the amount on the edgeNotQueue has to be positive at this point
            # This implies that vTimeLower <= time <= vTimeUpper + transitTime

            startFac = float(time-vTimeUpper)/transitTime if time > vTimeUpper else 0
            endFac = float(time-vTimeLower)/transitTime if time <= vTimeLower + transitTime else 1

            start = src + startFac * s
            end = src + endFac * s
            edge_pos.append((start, end))
            edge_colors.append(self.NTFColors[fk % len(self.NTFColors)])


        edge_pos = np.asarray(edge_pos)

        edge_colors = tuple([colorConverter.to_rgba(c, alpha=1)
                             for c in edge_colors])

        edgeCollection = LineCollection(edge_pos,
                                         colors=edge_colors,
                                         linewidths=self.tubeWidthSize,
                                         antialiaseds=(1,),
                                         transOffset=self.axes.transData,
                                         alpha = 1
                                         )

        edgeCollection.set_zorder(1)
        self.edgeColoring[edge] = edgeCollection
        self.axes.add_collection(edgeCollection)

    def draw_queue_color_box(self, edge, src, dst, p=0.25, radius=7, lastProportion=1):
        """
        Daw queue box with color of NTF
        :param edge: edge e=vw
        :param src: position of v
        :param dst: position of w
        :param p: length of box
        :param radius: radius/height of box
        :param lastProportion: start
        """
        if self.boxColoring[edge] is not None:
            self.boxColoring[edge].remove()
        self.boxColoring[edge] = None

        totalFlowOnQueue = sum(self.flowOnQueue[edge][fk][self.timePoints[self.currentTimeIndex]] for fk in range(len(self.nashFlow.flowIntervals)))

        if Utilities.is_eq_tol(totalFlowOnQueue, 0) or Utilities.is_eq_tol(self.maximalQueueSize, 0):
            return

        flowRatio = [max(0, float(self.flowOnQueue[edge][fk][self.timePoints[self.currentTimeIndex]])/totalFlowOnQueue)
                      for fk in range(len(self.nashFlow.flowIntervals))]
        totalRatio = totalFlowOnQueue/float(self.maximalQueueSize)

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

            lastProportion -= (totalRatio * flowRatio[fk])
        d = np.sqrt(np.sum(((dst - src) * p * (1-totalRatio)) ** 2))
        lastRec = Rectangle(src - delta,
                            width=d,
                            height=radius * 2,
                            transform=t,
                            facecolor='lightgrey',
                            linewidth=0,
                            alpha=1)
        boxes.append(lastRec)

        boxCollection = PatchCollection(boxes,
                                        match_original=True,
                                        antialiaseds=(1,),
                                        transOffset=self.axes.transData)
        self.boxColoring[edge] = boxCollection
        self.axes.add_collection(boxCollection)


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
        self.tubeWidthSize = smaller(self.tubeWidthSize)
        self.tube_collection.set_linewidth(self.tubeWidthSize)

        # Scale colored edges if existing
        for edge in self.network.edges():
            colorEdgeCollection = self.edgeColoring[edge]
            if colorEdgeCollection is not None:
                colorEdgeCollection.set_linewidth(self.tubeWidthSize)

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