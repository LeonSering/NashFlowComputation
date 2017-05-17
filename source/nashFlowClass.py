# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         nashFlowClass.py
# Description:  
# ===========================================================================
import networkx as nx
from collections import OrderedDict
from flowIntervalClass import FlowInterval

class NashFlow(nx.DiGraph):
    """description of class"""

    def __init__(self, nodes, edges, transitTimeDict, capacityDict, inflowRate):

        super(NashFlow, self).__init__()  # Call parents constructor

        self.add_nodes_from(nodes)  # Init nodes
        self.add_edges_from(edges)  # Init edges

        for edge in self.edges_iter():
            v, w = edge[0], edge[1]
            self[v][w]['transitTime'] = transitTimeDict[edge]
            self[v][w]['capacity'] = capacityDict[edge]

        
        self.inflowRate = inflowRate    # For the moment: constant
        self.restrictedFlowIntervals = []
        self.alphaToRestrictedFlowDict = OrderedDict()

        initialInterval = FlowInterval(self, lowerBoundTime=0, inflowRate=inflowRate)

    def node_label(node, time):
        pass



#Instance for debugging
inflowRate = 10
nodes = ['s', 'a', 'b', 't']
edges = [('s', 'a'), ('a', 'b'), ('a', 't'), ('b', 't')]
transitTimeList = [1, 1, 2, 1]
transitTimeDict = {entry[0]:entry[1] for entry in zip(edges, transitTimeList)}
capacityList = [5, 2, 7, 5]
capacityDict = {entry[0]:entry[1] for entry in zip(edges, capacityList)}

nf = NashFlow(nodes, edges, transitTimeDict, capacityDict, inflowRate)