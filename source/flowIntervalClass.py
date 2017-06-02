# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         flowIntervalClass.py
# Description:  
# ===========================================================================
import os
from itertools import combinations

from normalizedThinFlowClass import NormalizedThinFlow
from utilitiesClass import Utilities

TOL = 1e-8

class FlowInterval():
    """description of class"""

    def __init__(self, network, resettingEdges, lowerBoundTime, inflowRate, minCapacity, counter, outputDirectory, templateFile, scipFile):

        self.network = network
        self.resettingEdges = resettingEdges
        self.lowerBoundTime = lowerBoundTime
        self.upperBoundTime = None
        self.inflowRate = inflowRate
        self.minCapacity = minCapacity
        self.id         = counter
        self.outputDirectory = outputDirectory
        self.templateFile = templateFile
        self.scipFile = scipFile
        self.alpha      = None
        self.shortestPathNetwork = None # to be set from NashFlowClass
        self.numberOfSolvedIPs = 0
        self.NTFNodeLabelDict = {node:0 for node in self.network}
        self.NTFEdgeFlowDict = {edge:0 for edge in self.network.edges()}


        self.rootPath = os.path.join(self.outputDirectory, str(self.id) + '-FlowInterval-' + Utilities.get_time())
        Utilities.create_dir(self.rootPath)



    def compute_alpha(self, labelLowerBoundTimeDict):
        func = lambda v, w: (self.network[v][w]['transitTime'] + labelLowerBoundTimeDict[v] - labelLowerBoundTimeDict[w])\
                            /(self.NTFNodeLabelDict[w]-self.NTFNodeLabelDict[v])

        self.alpha = float('inf')

        for v,w in self.network.edges():
            e = (v,w)
            if e not in self.shortestPathNetwork.edges():
                if self.NTFNodeLabelDict[w] - self.NTFNodeLabelDict[v] > TOL:
                    self.alpha = min([self.alpha, func(v,w)])
            elif e in self.resettingEdges:
                if self.NTFNodeLabelDict[v] - self.NTFNodeLabelDict[w] > TOL:
                    self.alpha = min([self.alpha, func(v, w)])

        self.upperBoundTime = self.lowerBoundTime + self.alpha



    def get_NTF(self):

        self.counter = 0
        #self.naive_NTF_search() # Do not use, might lead to unwanted behaviour (i.e. could find solution to LP, even though E_0 was guessed badly -> result is not an NTF)
        self.backtrack_NTF_search_naive(remainingNodes=[v for v in self.shortestPathNetwork.nodes() if v != 's'], E_0=[])

        labels, flow = self.NTF.get_labels_and_flow()

        self.NTFNodeLabelDict.update(labels)
        self.NTFEdgeFlowDict.update(flow)

        self.assert_NTF()

    def backtrack_NTF_search_naive(self, remainingNodes, E_0):
        # Guarantees that f.a. nodes w there is at least one edge e=vw in E_0

        if not remainingNodes:
            NTF = NormalizedThinFlow(shortestPathNetwork=self.shortestPathNetwork, id=self.counter,
                                     resettingEdges=self.resettingEdges, flowEdges=E_0, inflowRate=self.inflowRate,
                                     minCapacity=self.minCapacity, outputDirectory=self.rootPath,
                                     templateFile=self.templateFile, scipFile=self.scipFile)
            self.numberOfSolvedIPs += 1
            if NTF.is_valid():
                self.NTF = NTF
                return True
            else:
                # Drop instance (necessary?)
                del NTF
                self.counter += 1
                return False

        w = remainingNodes[0]  # Node handled in this recursive call

        incomingEdges = self.shortestPathNetwork.in_edges(w)
        k = len(incomingEdges)

        while k>0:
            for partE_0 in combinations(incomingEdges, k):
                partE_0 = list(partE_0)

                recursiveCall = self.backtrack_NTF_search_naive(remainingNodes=remainingNodes[1:], E_0=E_0 + partE_0)
                if recursiveCall:
                    return True
            k -= 1

        return False


    def naive_NTF_search(self):
        found = False
        k = self.shortestPathNetwork.number_of_edges()
        counter = 0
        edges = self.shortestPathNetwork.edges()
        while k>0 and not found:
            for E_0 in combinations(edges, k):
                self.numberOfSolvedIPs += 1
                E_0 = list(E_0)
                NTF = NormalizedThinFlow(shortestPathNetwork=self.shortestPathNetwork, id=self.counter,
                                         resettingEdges=self.resettingEdges, flowEdges=E_0, inflowRate=self.inflowRate,
                                         minCapacity=self.minCapacity, outputDirectory=self.rootPath, templateFile=self.templateFile, scipFile=self.scipFile)

                if NTF.is_valid():
                    found = True
                    self.NTF = NTF
                    break
                else:
                    # Drop instance (necessary?)
                    del NTF

                self.counter += 1
            k -= 1

    def assert_NTF(self):
        # Works only on shortest path network!!
        p = lambda (v, w): max([self.NTFNodeLabelDict[v], self.NTFEdgeFlowDict[(v,w)]/self.network[v][w]['capacity']])\
                            if (v,w) not in self.resettingEdges \
                            else self.NTFEdgeFlowDict[(v,w)]/self.network[v][w]['capacity']
        for w in self.shortestPathNetwork:
            if self.shortestPathNetwork.in_edges(w):
                minimalCongestion = min(map(p, self.shortestPathNetwork.in_edges(w)))
                assert( Utilities.is_eq_tol(minimalCongestion, self.NTFNodeLabelDict[w]) )
        for v,w in self.shortestPathNetwork.edges():
            minimalCongestion = min(map(p, self.shortestPathNetwork.in_edges(w)))
            assert( Utilities.is_eq_tol(self.NTFEdgeFlowDict[v,w], 0) or Utilities.is_eq_tol(p((v,w)), minimalCongestion) )
