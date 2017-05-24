# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         utilitiesClass.py
# Description:
# ===========================================================================

import os
import time
import networkx as nx

class Utilities:

    @staticmethod
    def create_dir(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    @staticmethod
    def get_time():
        return time.strftime("%d_%m_%Y-%H_%M_%S", time.localtime())

    @staticmethod
    def is_eq_tol(a, b, tol=1e-8):
        return ( abs(a-b) <= tol )

    @staticmethod
    def is_not_eq_tol(a, b, tol=1e-8):
        return ( abs(a-b) > tol )

    @staticmethod
    def is_geq_tol(a, b, tol=1e-8):
        return ( a-b+tol >= 0 )


    @staticmethod
    def get_shortest_path_network(network, time, labels=None):
        shortestPathNetwork = None

        if not labels:
            # Use transit-times as edge weight
            labels = nx.single_source_dijkstra_path_length(G=network, source='s', weight='transitTime')    # Compute node distance from source

        # Create shortest path network containing _all_ shortest paths
        shortestPathEdges = [(edge[0], edge[1]) for edge in network.edges() if labels[edge[0]] + network[edge[0]][edge[1]]['transitTime'] <= labels[edge[1]]]
        shortestPathNetwork = nx.DiGraph()
        shortestPathNetwork.add_nodes_from(network)
        shortestPathNetwork.add_edges_from(shortestPathEdges)

        for edge in shortestPathEdges:
            v, w = edge[0], edge[1]
            shortestPathNetwork[v][w]['capacity'] = network[v][w]['capacity']
            shortestPathNetwork[v][w]['transitTime'] = network[v][w]['transitTime']

        for w in shortestPathNetwork:
            shortestPathNetwork.node[w]['dist'] = labels[w]
            shortestPathNetwork.node[w]['label'] = network.node[w]['label']
            shortestPathNetwork.node[w]['position'] = network.node[w]['position']

        return shortestPathNetwork


    @staticmethod
    def compute_min_capacity(network):
        minimumCapacity = float('inf')
        for edge in network.edges():
            v, w = edge[0], edge[1]
            minimumCapacity = min([minimumCapacity, network[v][w]['capacity']])

        return minimumCapacity

    @staticmethod
    def join_dicts(dict1, dict2):
        return {key:(dict1[key], dict2[key]) for key in dict1 if key in dict2}