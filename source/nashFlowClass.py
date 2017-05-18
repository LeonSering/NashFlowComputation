# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         nashFlowClass.py
# Description:  
# ===========================================================================
import networkx as nx
from collections import OrderedDict
from flowIntervalClass import FlowInterval

class NashFlow:
    """description of class"""

    def __init__(self, nodes, edges, transitTimeDict, capacityDict, inflowRate):

        self.network = nx.DiGraph()
        self.network.add_nodes_from(nodes)  # Init nodes
        self.network.add_edges_from(edges)  # Init edges

        for edge in self.network.edges_iter():
            v, w = edge[0], edge[1]
            self.network[v][w]['transitTime'] = transitTimeDict[edge]
            self.network[v][w]['capacity'] = capacityDict[edge]

        self.counter = 0
        self.inflowRate = inflowRate    # For the moment: constant
        self.restrictedFlowIntervals = []
        self.alphaToRestrictedFlowDict = OrderedDict()



        initialInterval = FlowInterval(self.network, resettingEdges=[], lowerBoundTime=0, inflowRate=inflowRate, counter=self.counter)
        self.counter += 1


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