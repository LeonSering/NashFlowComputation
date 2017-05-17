# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         flowIntervalClass.py
# Description:  
# ===========================================================================
from normalizedThinFlowClass import NormalizedThinFlow
import networkx as nx
class FlowInterval():
    """description of class"""

    def __init__(self, network, lowerBoundTime, inflowRate):
        self.network = network
        self.NTF = None
        self.lowerBoundTime = lowerBoundTime
        self.upperBoundTime = None
        self.inflowRate = inflowRate
        self.alpha      = None

        self.shortestPathNetwork = self.get_shortest_path_network(self.lowerBoundTime)
        print self.shortestPathNetwork.edges()

    def compute_alpha():
        pass # NTF needed


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