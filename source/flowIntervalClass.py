# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         flowIntervalClass.py
# Description:  
# ===========================================================================
from normalizedThinFlowClass import NormalizedThinFlow
import networkx as nx
from itertools import combinations
import os
from utilitiesClass import Utilities

ROOT_PATH = '/home/doc/Documents/Thesis/dev/NashFlowComputation/source/output'

class FlowInterval():
    """description of class"""

    def __init__(self, network, resettingEdges, lowerBoundTime, inflowRate, counter):

        self.network = network
        self.resettingEdges = resettingEdges
        self.lowerBoundTime = lowerBoundTime
        self.upperBoundTime = None
        self.inflowRate = inflowRate
        self.id         = counter
        self.alpha      = None
        self.nodeLabelDict = {node:0 for node in self.network}
        self.edgeFlowDict = {edge:0 for edge in self.network.edges()}


        self.outputDirectory = os.path.join(ROOT_PATH, str(self.id) + '-FlowInterval-' + Utilities.get_time())
        Utilities.create_dir(self.outputDirectory)

        self.shortestPathNetwork = self.get_shortest_path_network(self.lowerBoundTime)  # Compute shortest path network

        self.minCapacity = self.compute_min_capacity(self.shortestPathNetwork)

        self.get_NTF()
        #self.compute_alpha()

    def compute_alpha(self):
        pass # NTF needed

    def get_NTF(self):
        found = False
        k = self.shortestPathNetwork.number_of_edges()
        counter = 0
        edges = self.shortestPathNetwork.edges()
        while k>0 and not found:
            for E_0 in combinations(edges, k):
                E_0 = list(E_0)
                NTF = NormalizedThinFlow(shortestPathNetwork=self.shortestPathNetwork, id=counter, resettingEdges=self.resettingEdges, flowEdges=E_0, inflowRate=self.inflowRate, minCapacity=self.minCapacity, rootPath=self.outputDirectory)

                if NTF.is_valid():
                    found = True
                    self.NTF = NTF
                    break
                else:
                    # Drop instance (necessary?)
                    del NTF

                counter += 1
            k -= 1

        labels, flow = self.NTF.get_labels_and_flow()

        self.nodeLabelDict.update(labels)
        self.edgeFlowDict.update(flow)

        self.assert_NTF()

    def get_shortest_path_network(self, time):
        shortestPathNetwork = None
        if time != 0:
            # Get shortest network corresponding to node_labels (i.e. L)
            pass
        else:
            # Use transit-times as edge weight
            length = nx.single_source_dijkstra_path_length(G=self.network, source='s', weight='transitTime')    # Compute node distance from source

            # Create shortest path network containing _all_ shortest paths
            shortestPathEdges = [(edge[0], edge[1]) for edge in self.network.edges() if length[edge[0]] + self.network[edge[0]][edge[1]]['transitTime'] == length[edge[1]]]
            shortestPathNetwork = nx.DiGraph()
            shortestPathNetwork.add_nodes_from(self.network)
            shortestPathNetwork.add_edges_from(shortestPathEdges)

            for edge in shortestPathEdges:
                v, w = edge[0], edge[1]
                shortestPathNetwork[v][w]['capacity'] = self.network[v][w]['capacity']

        return shortestPathNetwork

    def compute_min_capacity(self, graph):
        minimumCapacity = float('inf')
        for edge in graph.edges():
            v, w = edge[0], edge[1]
            if self.network[v][w]['capacity'] < minimumCapacity:
                minimumCapacity = graph[v][w]['capacity']

        return minimumCapacity

    def assert_NTF(self):
        # Works only on shortest path network!!
        p = lambda (v,w): max([self.nodeLabelDict[v], self.edgeFlowDict[(v,w)]/self.network[v][w]['capacity']]) if (v,w) not in self.resettingEdges else self.edgeFlowDict[(v,w)]/self.network[v][w]['capacity']
        for w in self.shortestPathNetwork:
            if self.shortestPathNetwork.in_edges(w):
                minimalCongestion = min(map(p, self.shortestPathNetwork.in_edges(w)))
                assert( minimalCongestion == self.nodeLabelDict[w])
        for v,w in self.shortestPathNetwork.edges():
            minimalCongestion = min(map(p, self.shortestPathNetwork.in_edges(w)))
            assert( self.edgeFlowDict[v,w] == 0 or p((v,w)) == minimalCongestion)
