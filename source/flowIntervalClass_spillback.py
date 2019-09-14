# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         flowIntervalClass_spillback.py
# Description:  FlowInterval class managing one alpha-extension for Spillback Flows
# ===========================================================================

from source.flowIntervalClass import FlowInterval
from source.normalizedThinFlowClass_spillback import NormalizedThinFlow_spillback
from source.utilitiesClass import Utilities
from itertools import combinations

# =======================================================================================================================

TOL = 1e-8  # Tolerance


class FlowInterval_spillback(FlowInterval):
    """FlowInterval class managing spillback flows"""

    def __init__(self, network, resettingEdges, fullEdges, lowerBoundTime, inflowRate, minCapacity, counter,
                 outputDirectory,
                 templateFile, scipFile, timeout, minInflowBound):
        """
        :param network: Networkx Digraph instance
        :param resettingEdges: list of resetting edges
        :param fullEdges: list of full edges
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

        self.fullEdges = fullEdges
        self.NTFNodeSpillbackFactorDict = {node: 0 for node in self.network}
        self.minInflowBound = minInflowBound

    def set_minInflowBound(self, val):
        self.minInflowBound = val

    def transfer_inflowBound(self, network):
        # Transfers all entries of the form ['TFC']['inflowBound'] to ['inflowBound'] to assure compability
        for e in network.edges():
            (v, w) = e
            network[v][w]['inflowBound'] = network[v][w]['TFC']['inflowBound']

    def get_ntf(self):
        """Standard way to get sNTF. Uses preprocessing"""
        self.counter = 0
        graph, removedVertices = self.preprocessing()

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

        p = lambda e: max(
            [self.NTFNodeLabelDict[e[0]],
             self.NTFEdgeFlowDict[e] / (
                     self.network[e[0]][e[1]]['outCapacity'] * self.NTFNodeSpillbackFactorDict[e[1]])]) \
            if e not in self.resettingEdges \
            else self.NTFEdgeFlowDict[e] / (
                self.network[e[0]][e[1]]['outCapacity'] * self.NTFNodeSpillbackFactorDict[e[1]])
        xb = lambda e: float(self.NTFEdgeFlowDict[e]) / self.shortestPathNetwork[e[0]][e[1]]['inflowBound']

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

    def outflow_time_interval(self, u, v, t):
        """
        :param u: tail of edge
        :param v: head of edge
        :param t: timepoint
        :return: lowerBoundTime, upperBoundTime, j s.t. l_v(lowerBoundTime) <= t <= l_v(upperBoundTime) outflowBounds of [theta_j, theta_{j+1}[
        """
        j = 0
        for lowerBoundTime, upperBoundTime in self.network[u][v]['outflow']:
            if Utilities.is_geq_tol(t, lowerBoundTime) and Utilities.is_geq_tol(upperBoundTime, t):
                return lowerBoundTime, upperBoundTime, j
            j += 1

    def compute_alpha(self, labelLowerBoundTimeDict):
        """
        Compute alpha respecting the conditions
        :param labelLowerBoundTimeDict: Dict assigning lower bound times to nodes, i.e. v:l_v(\theta_k)
        """

        def get_max_alpha_conditions_2_3(u, v):
            return (self.network[u][v]['transitTime'] + labelLowerBoundTimeDict[u] -
                    labelLowerBoundTimeDict[v]) \
                   / (self.NTFNodeLabelDict[v] - self.NTFNodeLabelDict[u])

        def get_max_alpha_condition_4(u, v):
            """
            :param u: tail of edge
            :param v: head of edge
            :return: maximal alpha <= self.alpha satisfying constraint (4) for e = (u,v)
            """
            l_u = self.NTFNodeLabelDict[u]
            if Utilities.is_eq_tol(l_u, 0):
                # Constraint satisfied for all alpha <= self.alpha
                return self.alpha
            uTimeLower = labelLowerBoundTimeDict[u]
            _, upperBoundTime, j = self.outflow_time_interval(u, v, uTimeLower)

            if Utilities.is_eq_tol(uTimeLower, upperBoundTime):
                # Proceed with next interval, as intervals are actually half open, i.e. of form [.,.[
                j += 1

            intervalList = list(self.network[u][v]['outflow'].items())[j:]
            if not intervalList:
                # l_u(theta_k) = l_v(theta_k)
                return self.alpha

            intervalBounds, outflowVal = intervalList[0]
            _, upperBoundTime = intervalBounds
            if Utilities.is_greater_tol(outflowVal, self.network[u][v]['inCapacity']):
                # This edge is not of interest -> discard
                return self.alpha
            elif not Utilities.is_eq_tol(outflowVal, self.NTFEdgeFlowDict[(u, v)] / l_u):
                return self.alpha

            alpha = (upperBoundTime - uTimeLower) / l_u
            intervalList = intervalList[1:]
            for interval in intervalList:
                if alpha >= self.alpha:
                    # We could go higher, but previously computed alpha restricts us
                    return self.alpha

                outflowBounds, outflowVal_p = interval
                theta_p_lower, theta_p_upper = outflowBounds
                if not Utilities.is_eq_tol(outflowVal_p, outflowVal):
                    return min(alpha, self.alpha)
                else:
                    alpha += (theta_p_upper - theta_p_lower) / l_u
            if Utilities.is_eq_tol(self.NTFEdgeFlowDict[(u, v)] / self.NTFNodeLabelDict[v], outflowVal):
                alpha = float('inf')
            return min(alpha, self.alpha)

        def get_max_alpha_condition_5(u, v):
            """
            :param u: tail of edge
            :param v: head of edge
            :return: maximal alpha <= self.alpha satisfying constraint (5) for e = (u,v)
            """
            l_u = self.NTFNodeLabelDict[u]
            l_v = self.NTFNodeLabelDict[v]
            x_e = self.NTFEdgeFlowDict[(u, v)]
            sigma_e = self.network[u][v]['storage']
            if Utilities.is_eq_tol(l_u, 0):
                # Constraint satisfied for all alpha <= self.alpha
                return self.alpha
            uTimeLower = labelLowerBoundTimeDict[u]
            if self.lowerBoundTime == 0:
                # No other intervals have been computed yet
                vTimeLower = labelLowerBoundTimeDict[v]
                v_u_diff = vTimeLower - uTimeLower
                b = v_u_diff / l_u

                if Utilities.is_greater_tol(b * x_e, sigma_e):
                    # Edge becomes full before it even starts to push flow out again
                    return min(sigma_e / x_e, self.alpha)
                else:
                    D = x_e - l_u * (x_e / l_v)
                    if Utilities.is_leq_tol(D, 0):
                        return self.alpha
                    else:
                        R = sigma_e - b * x_e
                        return min(b + R / D, self.alpha)
            else:
                # There exist other intervals
                _, _, j = self.outflow_time_interval(u, v, uTimeLower)
                firstFlag = True
                intervalList = list(self.network[u][v]['outflow'].items())[j:]
                alpha = 0
                R = sigma_e - self.network[u][v]['load'][uTimeLower]
                for interval in intervalList:
                    if alpha >= self.alpha:
                        # We could go higher, but previously computed alpha restricts us
                        return self.alpha

                    intervalBounds, outflowVal_p = interval
                    theta_p_lower, theta_p_upper = intervalBounds

                    D = x_e - l_u * outflowVal_p
                    if firstFlag:
                        stepBound = (theta_p_upper - uTimeLower) / l_u
                        firstFlag = False
                    else:
                        stepBound = (theta_p_upper - theta_p_lower) / l_u

                    if not Utilities.is_leq_tol(D, 0) and Utilities.is_leq_tol(R / D, stepBound):
                        return min(alpha + R / D, self.alpha)
                    else:
                        alphaStep = stepBound
                        alpha += alphaStep

                    R -= alphaStep * D

                # Last interval should be checked as well
                D = x_e - l_u * (x_e / l_v)
                if Utilities.is_leq_tol(D, 0):
                    return self.alpha
                else:
                    alphaStep = R / D
                    alpha += alphaStep
                    return min(alpha, self.alpha)

        self.alpha = float('inf')
        for u, v in self.network.edges():
            e = (u, v)
            if e not in self.shortestPathNetwork.edges():  # Condition (3) in paper
                if self.NTFNodeLabelDict[v] - self.NTFNodeLabelDict[u] > TOL:
                    self.alpha = min(self.alpha, get_max_alpha_conditions_2_3(u, v))
            else:
                if e in self.resettingEdges:  # Condition (2) in paper
                    if self.NTFNodeLabelDict[u] - self.NTFNodeLabelDict[v] > TOL:
                        self.alpha = min(self.alpha, get_max_alpha_conditions_2_3(u, v))
                self.alpha = min(self.alpha, get_max_alpha_condition_5(u, v))  # Condition (5) in paper
                if e in self.fullEdges:
                    # Condition (4) in paper
                    self.alpha = min(self.alpha, get_max_alpha_condition_4(u, v))

        self.upperBoundTime = self.lowerBoundTime + self.alpha  # Set theta_{k+1}
