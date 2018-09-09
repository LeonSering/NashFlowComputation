# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         flowIntervalClass_spillback.py
# Description:  FlowInterval class managing one alpha-extension for Spillback Flows
# ===========================================================================

from flowIntervalClass import FlowInterval
from normalizedThinFlowClass_spillback import NormalizedThinFlow_spillback
from utilitiesClass import Utilities
from itertools import combinations

# =======================================================================================================================

TOL = 1e-8  # Tolerance


class FlowInterval_spillback(FlowInterval):
    """FlowInterval class managing spillback flows"""

    def __init__(self, network, resettingEdges, lowerBoundTime, inflowRate, minCapacity, counter, outputDirectory,
                 templateFile, scipFile, timeout):
        """
        :param network: Networkx Digraph instance
        :param resettingEdges: list of resetting edges
        :param lowerBoundTime: \theta_k
        :param inflowRate: u_0
        :param minCapacity: minimum capacity of all edges in network
        :param counter: unique ID of FlowInterval - needed for directory creation
        :param outputDirectory: directory to output scip logs
        :param templateFile: path of template which is used by SCIP
        :param scipFile: path to scip binary
        :param timeout: seconds until timeout. Deactivated if equal to 0
        """

        FlowInterval.__init__(self, network, resettingEdges, lowerBoundTime, inflowRate, minCapacity, counter,
                              outputDirectory,
                              templateFile, scipFile, timeout)

        self.NTFNodeSpillbackFactorDict = {node: 0 for node in self.network}

        self.minInflowBound = min([self.network[v][w]['inflowBound'] for (v, w) in self.network.edges()])

    def get_ntf(self):
        """Standard way to get sNTF. Uses preprocessing"""
        self.counter = 0
        graph, removedVertices = self.preprocessing()
        resettingEdges = [edge for edge in graph.edges() if edge in self.resettingEdges]

        graph = self.shortestPathNetwork
        resettingEdges = self.resettingEdges
        self.counter = 0

        self.backtrack_sNTF_search(remainingNodes=[v for v in graph.nodes() if v != 's'],
                                   E_0=[], graph=graph, resettingEdges=resettingEdges)

        labels, spillbackFactors, flow = self.NTF.get_labels_and_flow()

        labels, spillbackFactors = self.postprocessing(labels, spillbackFactors, removedVertices)

        self.NTFNodeLabelDict.update(labels)
        self.NTFEdgeFlowDict.update(flow)
        self.NTFNodeSpillbackFactorDict.update(spillbackFactors)

        self.assert_ntf()

    def backtrack_sNTF_search(self, remainingNodes, E_0, graph, resettingEdges):
        """
        sNTF Search: Ensures that f.a. nodes w there is at least one edge e=vw in E_0
        :param remainingNodes: List of nodes not handled yet
        :param E_0: list of selected edges
        :param graph: the graph
        :param resettingEdges: list of resetting edges of graph
        :return: True if NTF found, else False
        """

        # Check whether already aborted
        if self.aborted:
            return True

        if not remainingNodes:
            self.binaryVariableNumberList.append(len(E_0))
            self.NTF = NormalizedThinFlow_spillback(shortestPathNetwork=graph, id=self.counter,
                                                    resettingEdges=resettingEdges, flowEdges=E_0,
                                                    inflowRate=self.inflowRate,
                                                    minCapacity=self.minCapacity, outputDirectory=self.rootPath,
                                                    templateFile=self.templateFile, scipFile=self.scipFile,
                                                    minInflowBound=self.minInflowBound)
            self.NTF.run_order()
            self.numberOfSolvedIPs += 1
            if self.NTF.is_valid():
                self.foundNTF = True
                return True
            else:
                # Drop instance (necessary?)
                del self.NTF
                self.counter += 1
                return False

        w = remainingNodes[0]  # Node handled in this recursive call

        incomingEdges = graph.in_edges(w)
        n = len(incomingEdges)
        k = 1
        while k <= n:
            for partE_0 in combinations(incomingEdges, k):
                partE_0 = list(partE_0)

                recursiveCall = self.backtrack_sNTF_search(remainingNodes=remainingNodes[1:], E_0=E_0 + partE_0,
                                                           graph=graph, resettingEdges=resettingEdges)
                if recursiveCall:
                    return True
            k += 1

        return False

    def postprocessing(self, labels, spillbackFactors, missingVertices):
        """Update node labels for all missing vertices"""

        while missingVertices:
            w = missingVertices.pop()
            m = float('inf')
            for edge in self.shortestPathNetwork.in_edges(w):
                v = edge[0]
                if edge in self.resettingEdges:
                    m = 0
                    break
                else:
                    m = min(m, labels[v])
            labels[w] = m
            spillbackFactors[w] = 1

        return labels, spillbackFactors

    def assert_ntf(self):
        """Check if computed sNTF really is an sNTF"""
        # Works only on shortestPathNetwork
        for w in self.shortestPathNetwork:
            # Check if some spillback factor is not in ]0,1]
            assert (0 < self.NTFNodeSpillbackFactorDict[w] <= 1)

        p = lambda (v, w): max(
            [self.NTFNodeLabelDict[v],
             self.NTFEdgeFlowDict[(v, w)] / (self.network[v][w]['capacity'] * self.NTFNodeSpillbackFactorDict[w])]) \
            if (v, w) not in self.resettingEdges \
            else self.NTFEdgeFlowDict[(v, w)] / (self.network[v][w]['capacity'] * self.NTFNodeSpillbackFactorDict[w])
        xb = lambda (v, w): float(self.NTFEdgeFlowDict[(v, w)]) / self.network[v][w]['inflowBound']

        for w in self.shortestPathNetwork:
            if self.shortestPathNetwork.in_edges(w):
                minimalCongestion = min(map(p, self.shortestPathNetwork.in_edges(w)))
                assert (Utilities.is_eq_tol(minimalCongestion, self.NTFNodeLabelDict[w]))
            if self.shortestPathNetwork.out_edges(w):
                maximalCongestion = max(map(xb, self.shortestPathNetwork.out_edges(w)))
                assert (Utilities.is_geq_tol(self.NTFNodeLabelDict[w], maximalCongestion))
                if Utilities.is_greater_tol(1, self.NTFNodeSpillbackFactorDict[w]):
                    assert (Utilities.is_eq_tol(maximalCongestion, self.NTFNodeLabelDict[w]))
        for v, w in self.shortestPathNetwork.edges():
            minimalCongestion = min(map(p, self.shortestPathNetwork.in_edges(w)))
            assert (
                    Utilities.is_eq_tol(self.NTFEdgeFlowDict[v, w], 0) or Utilities.is_eq_tol(p((v, w)),
                                                                                              minimalCongestion))

        # Check if actually an s-t-flow
        for w in self.shortestPathNetwork:
            m = 0
            incomingEdges = self.shortestPathNetwork.in_edges(w)
            outgoingEdges = self.shortestPathNetwork.out_edges(w)
            for e in incomingEdges:
                m += self.NTFEdgeFlowDict[e]
            for e in outgoingEdges:
                m -= self.NTFEdgeFlowDict[e]

            if w == 's':
                assert (Utilities.is_eq_tol(m, (-1) * self.inflowRate))
            elif w == 't':
                assert (Utilities.is_eq_tol(m, self.inflowRate))
            else:
                assert (Utilities.is_eq_tol(m, 0))
