# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         nashFlowClass_spillback.py
# Description:  Class NashFlow_spillback maintains list of FlowInterval instances; coordinates computation of dynamic equilibrium
# ===========================================================================

import os
import time
from collections import OrderedDict

from .flowIntervalClass_spillback import FlowInterval_spillback
from .nashFlowClass import NashFlow
from .utilitiesClass import Utilities

# ======================================================================================================================

TOL = 1e-8  # Tolerance


class NashFlow_spillback(NashFlow):
    """Maintains Nash Flow over time for the spillback case"""

    def __init__(self, graph, inflowRate, numberOfIntervals, outputDirectory, templateFile, scipFile, cleanUpBool,
                 timeout):
        """
        :param graph: Networkx Digraph instance
        :param inflowRate: u_0
        :param numberOfIntervals: number of intervals that will be computed. -1 if all
        :param outputDirectory: path where output should be saved
        :param templateFile: Selected method, i.e. 0,1,2
        :param scipFile: path to scip binary
        :param cleanUpBool: If true, then cleanup
        :param timeout: seconds until timeout. Deactivated if equal to 0
        """

        self.network = graph.copy()
        self.inflowRate = inflowRate  # For the moment: constant
        self.numberOfIntervals = numberOfIntervals  # No. of intervals to compute
        self.outputDirectory = outputDirectory

        # Template File from /source/templates
        self.templateFile = os.path.join(os.getcwd(), 'source', 'templates',
                                         'algorithm_spillback_' + str(templateFile + 1) + '.zpl')

        self.scipFile = scipFile
        self.cleanUpBool = cleanUpBool
        self.numberOfSolvedIPs = 0
        self.computationalTime = 0
        self.infinityReached = False  # True if last interval has alpha = +inf
        self.timeout = timeout

        self.minOutCapacity = Utilities.compute_min_attr_of_network(network=self.network, attr='outCapacity')
        self.counter = 0
        self.preprocessedNodes = 0
        self.preprocessedEdges = 0

        # Create directory for Nash-Flow
        self.rootPath = os.path.join(self.outputDirectory, 'NashFlow_spillback-' + Utilities.get_time())
        Utilities.create_dir(self.rootPath)

        self.flowIntervals = []  # List containing triplets of form (lowerBound, upperBound, FlowInterval_spillback-instance)
        self.lowerBoundsToIntervalDict = OrderedDict()
        self.animationIntervals = {edge: [] for edge in self.network.edges()}

    def compute_flowInterval(self):
        """Method to compute a single flowInterval"""
        # Get lowerBoundTime
        lowerBoundTime = 0 if not self.flowIntervals else self.flowIntervals[-1][1]
        # Compute resettingEdges
        cmp_queue = lambda v, w, t: \
            Utilities.is_greater_tol(self.node_label(w, t),
                                     self.node_label(v, t) + self.network[v][w]['transitTime'])

        resettingEdges = [(v, w) for v, w in self.network.edges() if cmp_queue(v, w, lowerBoundTime)] \
            if lowerBoundTime > 0 else []

        edges_to_choose_from = resettingEdges   # Every full edge must be a resettingEdge
        fullEdges = [(v, w) for v, w in edges_to_choose_from if
                     self.is_full(v, w, self.node_label(v, lowerBoundTime))] if lowerBoundTime > 0 else []


        minInflowBound = None
        interval = FlowInterval_spillback(self.network, resettingEdges=resettingEdges, fullEdges=fullEdges,
                                          lowerBoundTime=lowerBoundTime,
                                          inflowRate=self.inflowRate, minCapacity=self.minOutCapacity,
                                          counter=self.counter,
                                          outputDirectory=self.rootPath, templateFile=self.templateFile,
                                          scipFile=self.scipFile,
                                          timeout=self.timeout, minInflowBound=minInflowBound)

        if lowerBoundTime == 0:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(
                self.network)  # Compute shortest path network
            for v, w in interval.shortestPathNetwork.edges():
                interval.shortestPathNetwork[v][w]['inflowBound'] = self.network[v][w]['inCapacity']
        else:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(self.network, labels={
                v: self.node_label(v, lowerBoundTime) for v in self.network})  # Compute shortest path network

            for v, w in interval.shortestPathNetwork.edges():
                vTimeLower = self.node_label(v, lowerBoundTime)
                minimizer = self.get_outflow(v, w, vTimeLower) if (v, w) in fullEdges else float('inf')
                interval.shortestPathNetwork[v][w]['inflowBound'] = min(minimizer, self.network[v][w]['inCapacity'])

        minInflowBound = Utilities.compute_min_attr_of_network(interval.shortestPathNetwork, 'inflowBound')
        interval.set_minInflowBound(minInflowBound)

        start = time.time()
        interval.get_ntf()
        '''
        if self.advancedAlgo:
            interval.get_ntf_advanced()
        else:
            interval.get_ntf()
        '''
        end = time.time()
        self.computationalTime += (end - start)
        interval.computationalTime = end - start
        self.preprocessedNodes += interval.preprocessedNodes
        self.preprocessedEdges += interval.preprocessedEdges
        self.lowerBoundsToIntervalDict[lowerBoundTime] = interval

        if lowerBoundTime == 0:
            self.init_edge_properties()

        interval.compute_alpha({node: self.node_label(node, lowerBoundTime) for node in self.network})
        self.flowIntervals.append((interval.lowerBoundTime, interval.upperBoundTime, interval))

        # Update in/out-flow rates
        self.update_edge_properties(lowerBoundTime, interval)

        self.counter += 1
        self.numberOfSolvedIPs += interval.numberOfSolvedIPs

        self.infinityReached = (interval.alpha == float('inf'))

    def init_edge_properties(self):
        # init in/outflow
        for v, w in self.network.edges():
            vTimeLower = self.node_label(v, 0)
            wTimeLower = self.node_label(w, 0)

            self.network[v][w]['inflow'] = OrderedDict()
            self.network[v][w]['inflow'][(0, vTimeLower)] = 0

            self.network[v][w]['outflow'] = OrderedDict()
            self.network[v][w]['outflow'][(0, wTimeLower)] = 0

            self.network[v][w]['cumulativeInflow'] = OrderedDict()
            self.network[v][w]['cumulativeInflow'][0] = 0
            self.network[v][w]['cumulativeInflow'][vTimeLower] = 0

            self.network[v][w]['cumulativeOutflow'] = OrderedDict()
            self.network[v][w]['cumulativeOutflow'][0] = 0
            self.network[v][w]['cumulativeOutflow'][wTimeLower] = 0

            self.network[v][w]['queueSize'] = OrderedDict()
            self.network[v][w]['queueSize'][0] = 0
            self.network[v][w]['queueSize'][vTimeLower + self.network[v][w]['transitTime']] = 0

            self.network[v][w]['load'] = OrderedDict()
            self.network[v][w]['load'][0] = 0
            self.network[v][w]['load'][vTimeLower] = 0

    def update_edge_properties(self, lowerBoundTime, interval):
        """
        Updates the edge properties after computation of a new flowInterval
        :param lowerBoundTime: lowerBoundTime of flowInterval
        :param interval: flowInterval
        """

        for v, w in self.network.edges():
            # Inflow changes
            vTimeLower, vTimeUpper = self.node_label(v, interval.lowerBoundTime), self.node_label(v,
                                                                                                  interval.upperBoundTime)
            inflowChangeBool = Utilities.is_not_eq_tol(interval.NTFNodeLabelDict[v],
                                                       0)  # Can we extend the inflow interval?
            inflowVal = interval.NTFEdgeFlowDict[(v, w)] / interval.NTFNodeLabelDict[v] if inflowChangeBool else 0
            if inflowChangeBool:
                self.network[v][w]['inflow'][(vTimeLower, vTimeUpper)] = inflowVal

            if vTimeUpper < float('inf'):
                vLastTime = next(reversed(self.network[v][w]['cumulativeInflow']))
                self.network[v][w]['cumulativeInflow'][vTimeUpper] = self.network[v][w]['cumulativeInflow'][
                                                                         vLastTime] + inflowVal * (
                                                                             vTimeUpper - vTimeLower)

            # Outflow changes
            wTimeLower, wTimeUpper = self.node_label(w, interval.lowerBoundTime), self.node_label(w,
                                                                                                  interval.upperBoundTime)
            outflowChangeBool = Utilities.is_not_eq_tol(interval.NTFNodeLabelDict[w],
                                                        0)  # Can we extend the outflow interval?
            outflowVal = interval.NTFEdgeFlowDict[(v, w)] / interval.NTFNodeLabelDict[w] if outflowChangeBool else 0
            if outflowChangeBool:
                self.network[v][w]['outflow'][(wTimeLower, wTimeUpper)] = outflowVal

            if wTimeUpper < float('inf'):
                wLastTime = next(reversed(self.network[v][w]['cumulativeOutflow']))
                self.network[v][w]['cumulativeOutflow'][wTimeUpper] = self.network[v][w]['cumulativeOutflow'][
                                                                          wLastTime] + outflowVal * (
                                                                              wTimeUpper - wTimeLower)

            # Queue size changes
            if vTimeUpper < float('inf'):
                self.update_queue_size(v, w, vTimeLower, vTimeUpper)
            self.animationIntervals[(v, w)].append(((vTimeLower, vTimeUpper), (wTimeLower, wTimeUpper)))

            if vTimeUpper <= wTimeUpper and vTimeUpper != float('inf'):
                # Lies on shortest path
                self.network[v][w]['load'][vTimeUpper] = self.arc_load(v, w, vTimeUpper)

    def update_queue_size(self, v, w, vTimeLower, vTimeUpper):
        # This is now more complicated, as outflow changes even if queue exists (due to spillback)
        transitTime = self.network[v][w]['transitTime']
        lastQueueSizeTime = next(reversed(self.network[v][w]['queueSize']))  # Should be vTimeLower + transitTime
        lastQueueSize = self.network[v][w]['queueSize'][lastQueueSizeTime]
        inflowVal = self.network[v][w]['inflow'][(vTimeLower, vTimeUpper)]

        # Find all changes in outflow in interval [vTimeLower + transitTime, vTimeUpper + transitTime]
        l, u = vTimeLower + transitTime, vTimeUpper + transitTime
        lastSize = lastQueueSize
        for timeInterval, outflowVal in self.network[v][w]['outflow'].items():
            wTimeLower, wTimeUpper = timeInterval
            if wTimeUpper <= l:
                # Not relevant
                continue
            elif u < wTimeLower:
                # Not relevant
                break
            elif l <= wTimeUpper <= u:
                lastSize = max(0, lastSize + (inflowVal - outflowVal) * (wTimeUpper - l))
                l = wTimeUpper
                self.network[v][w]['queueSize'][l] = lastSize
            elif l <= u < wTimeUpper:
                lastSize = max(0, lastSize + (inflowVal - outflowVal) * (u - l))
                l = u
                self.network[v][w]['queueSize'][l] = lastSize
                break

    def get_cumulative_inflow(self, v, w, t):
        """
        :param v: tail of edge
        :param w: head of edge
        :param t: time
        :return: F_(v,w)^+(t)
        """
        if Utilities.is_leq_tol(t, 0):
            return 0
        for timeInterval, inflowVal in reversed(self.network[v][w]['inflow'].items()):
            vTimeLower, vTimeUpper = timeInterval
            if Utilities.is_between_tol(vTimeLower, t, vTimeUpper):
                # This is the interval in which t lies
                return self.network[v][w]['cumulativeInflow'][vTimeLower] + inflowVal * (t - vTimeLower)

    def get_cumulative_outflow(self, v, w, t):
        """
        :param v: tail of edge
        :param w: head of edge
        :param t: time
        :return: F_(v,w)^-(t)
        """
        if Utilities.is_leq_tol(t, 0):
            return 0
        for timeInterval, outflowVal in reversed(self.network[v][w]['outflow'].items()):
            wTimeLower, wTimeUpper = timeInterval
            if Utilities.is_between_tol(wTimeLower, t, wTimeUpper):
                # This is the interval in which t lies
                return self.network[v][w]['cumulativeOutflow'][wTimeLower] + outflowVal * (t - wTimeLower)

    def arc_load(self, v, w, t):
        """
        Equivalent to d_(v,w)(t)
        :param v: node
        :param w: node (s.t. (v,w) is an edge)
        :param t: float, time
        :return: d_(v,w)(t)
        """
        return self.get_cumulative_inflow(v, w, t) - self.get_cumulative_outflow(v, w, t)

    def is_full(self, v, w, t):
        """
        :param v: tail of edge
        :param w: head of edge
        :param t: time
        :return: True if d_(v,w)(t) ~= storage(v,w)
        """
        # TODO: SHITTY WORKAROUND
        try:
            load = self.arc_load(v, w, t)
            return Utilities.is_eq_tol(load, self.network[v][w]['storage'])
        except TypeError:
            return False
