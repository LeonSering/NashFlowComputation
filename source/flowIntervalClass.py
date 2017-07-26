# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         flowIntervalClass.py
# Description:  
# ===========================================================================
import os
from itertools import combinations
from collections import deque
from normalizedThinFlowClass import NormalizedThinFlow
from utilitiesClass import Utilities
import threading
import time
import signal
import subprocess
TOL = 1e-8


class FlowInterval:
    """description of class"""

    def __init__(self, network, resettingEdges, lowerBoundTime, inflowRate, minCapacity, counter, outputDirectory,
                 templateFile, scipFile, timeout):

        self.network = network
        self.resettingEdges = resettingEdges
        self.lowerBoundTime = lowerBoundTime
        self.upperBoundTime = None
        self.inflowRate = inflowRate
        self.minCapacity = minCapacity
        self.id = counter
        self.outputDirectory = outputDirectory
        self.templateFile = templateFile
        self.scipFile = scipFile
        self.timeout = timeout


        self.foundNTF = False
        self.aborted = False
        self.alpha = None
        self.shortestPathNetwork = None  # to be set from NashFlowClass
        self.numberOfSolvedIPs = 0
        self.computationalTime = -1
        self.preprocessedNodes = 0
        self.preprocessedEdges = 0
        self.NTFNodeLabelDict = {node: 0 for node in self.network}
        self.NTFEdgeFlowDict = {edge: 0 for edge in self.network.edges()}


        self.rootPath = os.path.join(self.outputDirectory, str(self.id) + '-FlowInterval-' + Utilities.get_time())
        Utilities.create_dir(self.rootPath)

        if self.timeout > 0:
            self.timeoutThread = threading.Thread(target=self.timeout_control)
            self.timeoutThread.start()

    def timeout_control(self):
        startTime = time.time()
        while not self.foundNTF and time.time() - startTime <= self.timeout:
            time.sleep(1)

        if not self.foundNTF:
            self.aborted = True
            out = subprocess.check_output(['ps', '-A'])
            for line in out.splitlines():
                if "scip" in line:
                    print "KILLING SCIP PROCESS"
                    pid = int(line.split(None, 1)[0])
                    os.kill(pid, signal.SIGKILL)



    def compute_alpha(self, labelLowerBoundTimeDict):
        func = lambda v, w: (
                                self.network[v][w]['transitTime'] + labelLowerBoundTimeDict[v] -
                                labelLowerBoundTimeDict[w]) \
                            / (self.NTFNodeLabelDict[w] - self.NTFNodeLabelDict[v])

        self.alpha = float('inf')

        for v, w in self.network.edges():
            e = (v, w)
            if e not in self.shortestPathNetwork.edges():
                if self.NTFNodeLabelDict[w] - self.NTFNodeLabelDict[v] > TOL:
                    self.alpha = min([self.alpha, func(v, w)])
            elif e in self.resettingEdges:
                if self.NTFNodeLabelDict[v] - self.NTFNodeLabelDict[w] > TOL:
                    self.alpha = min([self.alpha, func(v, w)])

        self.upperBoundTime = self.lowerBoundTime + self.alpha

    def get_NTF(self):
        USE_PREPROCESSING = False
        graph = self.shortestPathNetwork
        resettingEdges = self.resettingEdges
        self.counter = 0
        # self.naive_NTF_search() # Do not use, might lead to unwanted behaviour (i.e. could find solution to LP, even though E_0 was guessed badly -> result is not an NTF)
        if USE_PREPROCESSING:
            graph, removedVertices = self.preprocessing()
            resettingEdges = [edge for edge in graph.edges() if edge in self.resettingEdges]

        self.backtrack_NTF_search_naive(remainingNodes=[v for v in graph.nodes() if v != 's'],
                                        E_0=[], graph=graph, resettingEdges=resettingEdges)

        labels, flow = self.NTF.get_labels_and_flow()

        if USE_PREPROCESSING:
            labels = self.postprocessing(labels, removedVertices)


        self.NTFNodeLabelDict.update(labels)
        self.NTFEdgeFlowDict.update(flow)

        self.assert_NTF()

    def get_NTF_advanced(self):

        USE_PREPROCESSING = True
        self.counter = 0
        graph = self.shortestPathNetwork
        E_0 = []
        if USE_PREPROCESSING:
            graph, removedVertices = self.preprocessing()
            # All resetting edges have to be part of E_0
            E_0 = [e for e in self.resettingEdges if e in graph.edges()]  # Init


        isolationDict = dict()
        violatedNodes = []
        remainingEdges = [e for e in graph.edges() if e not in E_0]
        for v in graph.nodes():
            foundOut, foundIn = False, False
            for edge in graph.out_edges(v):
                if edge in E_0:
                    foundOut = True
                    break
            for edge in graph.in_edges(v):
                if edge in E_0:
                    foundIn = True
                    break

            if v == 's':
                foundIn = True
            elif v == 't':
                foundOut = True

            isolationDict[v] = (foundIn, foundOut)
            if foundIn != foundOut:
                violatedNodes.append(v)

        if USE_PREPROCESSING:
            self.backtrack_NTF_search_advanced(E_0, isolationDict, violatedNodes, remainingEdges, graph, E_0)
        else:
            self.backtrack_NTF_search_advanced(E_0, isolationDict, violatedNodes, remainingEdges, graph, self.resettingEdges)

        labels, flow = self.NTF.get_labels_and_flow()

        if USE_PREPROCESSING:
            labels = self.postprocessing(labels, removedVertices)
        self.NTFNodeLabelDict.update(labels)
        self.NTFEdgeFlowDict.update(flow)

        self.assert_NTF()

    def violatedTuple(self, t):
        return t[0] != t[1]

    def backtrack_NTF_search_advanced(self, E_0, isolationDict, violatedNodes, remainingEdges, graph, resettingEdges):
        # Ensures that each node lies on some path s-t-path in E_0 or is isolated

        # Check whether already aborted
        if self.aborted:
            return True

        if not violatedNodes:
            self.NTF = NormalizedThinFlow(shortestPathNetwork=graph, id=self.counter,
                                     resettingEdges=resettingEdges, flowEdges=E_0, inflowRate=self.inflowRate,
                                     minCapacity=self.minCapacity, outputDirectory=self.rootPath,
                                     templateFile=self.templateFile, scipFile=self.scipFile)
            self.NTF.run_order()
            self.numberOfSolvedIPs += 1
            if self.NTF.is_valid():
                self.foundNTF = True
                return True
            else:
                # Drop instance (necessary?)
                del self.NTF
                self.counter += 1

                if remainingEdges:
                    n = len(remainingEdges)
                    k = 1
                    while k <= n:
                        for selectedRemainers in combinations(remainingEdges, k):
                            selectedRemainers = list(selectedRemainers)
                            isolationDictCopy = dict(isolationDict)
                            for edge in selectedRemainers:
                                v, w = edge
                                isolationDictCopy[v] = (isolationDictCopy[v][0], True)
                                isolationDictCopy[w] = (True, isolationDictCopy[w][1])

                            recursiveCall = self.backtrack_NTF_search_advanced(E_0 + selectedRemainers, isolationDictCopy,
                                                                          [v for v in graph.nodes() if
                                                                           self.violatedTuple(isolationDictCopy[v])],
                                                                          [e for e in remainingEdges if e not in selectedRemainers], graph, resettingEdges)

                            if recursiveCall:
                                return True
                        k += 1

                return False

        w = violatedNodes[0]  # Node handled in this recursive call
        if isolationDict[w][0]:
            # No outgoing edges but incoming edges
            isolationDict[w] = (True, True)
            remainingOutgoing = [e for e in graph.out_edges(w) if e in remainingEdges]
            n = len(remainingOutgoing)
            k = 1
            while k <= n:
                for selectedRemainers in combinations(remainingOutgoing, k):
                    selectedRemainers = list(selectedRemainers)
                    isolationDictCopy = dict(isolationDict)



                    for edge in selectedRemainers:
                        v = edge[1] # Edge from w to v
                        isolationDictCopy[v] = (True, isolationDictCopy[v][1])



                    recursiveCall = self.backtrack_NTF_search_advanced(E_0 + selectedRemainers, isolationDictCopy,
                                                                       [v for v in graph.nodes() if
                                                                        self.violatedTuple(isolationDictCopy[v])],
                                                                       [e for e in remainingEdges if
                                                                        e not in selectedRemainers], graph,
                                                                       resettingEdges)

                    if recursiveCall:
                        return True
                k += 1

        elif isolationDict[w][1]:
            # No incoming edges but outgoing edges
            isolationDict[w] = (True, True)
            remainingIncoming = [e for e in graph.in_edges(w) if e in remainingEdges]
            n = len(remainingIncoming)
            k = 1
            while k <= n:
                for selectedRemainers in combinations(remainingIncoming, k):
                    selectedRemainers = list(selectedRemainers)
                    isolationDictCopy = dict(isolationDict)
                    for edge in selectedRemainers:
                        v = edge[0] # Edge from v to w
                        isolationDictCopy[v] = (isolationDictCopy[v][0], True)

                    recursiveCall = self.backtrack_NTF_search_advanced(E_0 + selectedRemainers, isolationDictCopy,
                                                                       [v for v in graph.nodes() if
                                                                        self.violatedTuple(isolationDictCopy[v])],
                                                                       [e for e in remainingEdges if
                                                                        e not in selectedRemainers], graph,
                                                                       resettingEdges)
                    if recursiveCall:
                        return True
                k += 1

        return False


    def preprocessing(self):
        '''Iteratively remove all nodes that have no outgoing edges (except sink t)'''
        graph = self.shortestPathNetwork.copy()

        queue = deque([w for w in graph.nodes()
                 if w != 't' and graph.out_degree(w) == 0])

        removedVertices = deque([])

        while queue:
            w = queue.pop()
            for edge in graph.in_edges(w):
                v = edge[0]
                if graph.out_degree(v) == 1 and v != 't':
                    queue.append(v)
            graph.remove_node(w)
            removedVertices.append(w)

        self.preprocessedNodes = len(removedVertices)
        self.preprocessedEdges = self.shortestPathNetwork.number_of_edges() - graph.number_of_edges()

        return graph, removedVertices

    def postprocessing(self, labels, missingVertices):
        '''Update node labels for all missing vertices'''

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

        return labels



    def backtrack_NTF_search_naive(self, remainingNodes, E_0, graph, resettingEdges):
        # Guarantees that f.a. nodes w there is at least one edge e=vw in E_0

        # Check whether already aborted
        if self.aborted:
            return True

        if not remainingNodes:

            self.NTF = NormalizedThinFlow(shortestPathNetwork=graph, id=self.counter,
                                     resettingEdges=resettingEdges, flowEdges=E_0, inflowRate=self.inflowRate,
                                     minCapacity=self.minCapacity, outputDirectory=self.rootPath,
                                     templateFile=self.templateFile, scipFile=self.scipFile)
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
        k = len(incomingEdges)

        while k > 0:
            for partE_0 in combinations(incomingEdges, k):
                partE_0 = list(partE_0)

                recursiveCall = self.backtrack_NTF_search_naive(remainingNodes=remainingNodes[1:], E_0=E_0 + partE_0, graph=graph, resettingEdges=resettingEdges)
                if recursiveCall:
                    return True
            k -= 1

        return False

    def naive_NTF_search(self):



        found = False
        k = self.shortestPathNetwork.number_of_edges()
        edges = self.shortestPathNetwork.edges()
        while k > 0 and not found:
            for E_0 in combinations(edges, k):
                # Check whether already aborted
                if self.aborted:
                    return True

                self.numberOfSolvedIPs += 1
                E_0 = list(E_0)
                self.NTF = NormalizedThinFlow(shortestPathNetwork=self.shortestPathNetwork, id=self.counter,
                                         resettingEdges=self.resettingEdges, flowEdges=E_0, inflowRate=self.inflowRate,
                                         minCapacity=self.minCapacity, outputDirectory=self.rootPath,
                                         templateFile=self.templateFile, scipFile=self.scipFile)
                self.NTF.run_order()
                if self.NTF.is_valid():
                    found = True
                    self.foundNTF = True
                    break
                else:
                    # Drop instance (necessary?)
                    del self.NTF

                self.counter += 1
            k -= 1

    def assert_NTF(self):
        # Works only on shortest path network!!
        p = lambda (v, w): max(
            [self.NTFNodeLabelDict[v], self.NTFEdgeFlowDict[(v, w)] / self.network[v][w]['capacity']]) \
            if (v, w) not in self.resettingEdges \
            else self.NTFEdgeFlowDict[(v, w)] / self.network[v][w]['capacity']
        for w in self.shortestPathNetwork:
            if self.shortestPathNetwork.in_edges(w):
                minimalCongestion = min(map(p, self.shortestPathNetwork.in_edges(w)))
                assert (Utilities.is_eq_tol(minimalCongestion, self.NTFNodeLabelDict[w]))
        for v, w in self.shortestPathNetwork.edges():
            minimalCongestion = min(map(p, self.shortestPathNetwork.in_edges(w)))
            assert (
                Utilities.is_eq_tol(self.NTFEdgeFlowDict[v, w], 0) or Utilities.is_eq_tol(p((v, w)), minimalCongestion))

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
