# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         plotQueueCanvasClass.py
# Description:  Class to plot queues
# ===========================================================================

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import figure
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from source.utilitiesClass import Utilities

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ======================================================================================================================

class PlotQueueCanvas(FigureCanvas):
    """Class to plot queues"""

    def __init__(self):
        self.figure = figure.Figure()
        stretchFactor = 0.1
        self.Xlim = (stretchFactor * (-100), stretchFactor * 100)
        self.Ylim = (0, 100)
        super(PlotQueueCanvas, self).__init__(self.figure)  # Call parents constructor

        self.network = nx.DiGraph()  # Has no relation to graph we're working on.
        self.src, self.dst = (0, 90), (0, 10)
        self.network.add_nodes_from(
            [('s', {'position': self.src, 'label': 's'}), ('t', {'position': self.dst, 'label': 't'})])
        self.network.add_edge('s', 't')
        self.focusEdge = None
        self.edgeWidthSize = 4
        self.boxColoring = None
        self.init_plot()

    def change_focusEdge(self, v=None, w=None):
        """Change currently focussed edge to (v,w)"""
        if not v or not w:
            # No edge given, hence node must be selected
            self.focusEdge = None
            if self.boxColoring is not None:
                self.boxColoring.remove()
                self.boxColoring = None
            self.draw_idle()
        else:
            self.focusEdge = (v, w)
            self.update_queue_animation()
            self.draw_idle()

    def update_information_callback(self, callback):
        """Precomputed information has updated, hence new information available"""
        attributeList = ['nashFlow', 'timePoints', 'flowOnQueue', 'maxQueueSize', 'NTFColors', 'currentTimeIndex']
        for attrName in attributeList:
            obj = getattr(callback, attrName)
            setattr(self, attrName, obj)

    def time_changed(self, sliderVal):
        """
        Update the animation given a time index
        :param sliderVal: index
        """
        self.currentTimeIndex = sliderVal
        self.update_queue_animation()
        self.draw_idle()

    def update_queue_animation(self, radius=7):
        """Update queue animation to display different edge queue"""
        if not self.focusEdge:
            return
        if self.boxColoring:
            self.boxColoring.remove()
        self.boxColoring = None

        # Work setting
        edge = self.focusEdge
        time = self.timePoints[self.currentTimeIndex]

        totalFlowOnQueue = sum(self.flowOnQueue[edge][fk][time] for fk in range(len(self.nashFlow.flowIntervals)))

        if Utilities.is_eq_tol(totalFlowOnQueue, 0) or Utilities.is_eq_tol(self.maxQueueSize, 0):
            return

        flowRatio = [max(0, float(self.flowOnQueue[edge][fk][time]) / totalFlowOnQueue)
                     for fk in range(len(self.nashFlow.flowIntervals))]
        totalRatio = totalFlowOnQueue / float(self.maxQueueSize)

        delta = np.array([0, radius])
        src = np.array(self.src)
        dst = np.array(self.dst)
        s = dst - src
        angle = np.rad2deg(np.arctan2(s[1], s[0]))
        t = matplotlib.transforms.Affine2D().rotate_deg_around(src[0], src[1], angle)
        boxes = []
        lastProportion = 1
        for fk in range(len(self.nashFlow.flowIntervals)):
            if Utilities.is_eq_tol(flowRatio[fk], 0):
                continue

            d = np.sqrt(np.sum(((dst - src) * lastProportion) ** 2))
            rec = Rectangle(src - delta,
                            width=d,
                            height=radius * 2,
                            transform=t,
                            facecolor=self.NTFColors[fk % len(self.NTFColors)],
                            linewidth=int(lastProportion),
                            alpha=1)
            boxes.append(rec)

            lastProportion -= (totalRatio * flowRatio[fk])
        d = np.sqrt(np.sum(((dst - src) * (1 - totalRatio)) ** 2))
        lastRec = Rectangle(src - delta,
                            width=d,
                            height=radius * 2,
                            transform=t,
                            facecolor='white',
                            linewidth=0,
                            alpha=1)
        boxes.append(lastRec)

        boxCollection = PatchCollection(boxes,
                                        match_original=True,
                                        antialiaseds=(1,),
                                        transOffset=self.axes.transData)
        self.boxColoring = boxCollection
        self.axes.add_collection(boxCollection)

    def init_plot(self):
        """
        Update canvas to plot new queue visualization
        """
        self.figure.clf()  # Clear current figure window
        self.axes = self.figure.add_axes([0, 0, 1, 1])
        # self.axes.set_aspect("equal")
        self.axes.set_xlim(self.Xlim)
        self.axes.set_ylim(self.Ylim)

        self.axes.axis('off')  # Hide axes in the plot

        positions = nx.get_node_attributes(self.network, 'position')

        # Plot Queue
        color = ['black']
        self.bgBox = self.draw_edges(self.network, pos=positions, ax=self.axes,
                                     arrow=True,
                                     edge_color=color, width=self.edgeWidthSize)
        self.draw_idle()

    @staticmethod
    def draw_edges(G, pos,
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

        return boxes
