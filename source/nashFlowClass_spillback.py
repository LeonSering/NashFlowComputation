# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         nashFlowClass_spillback.py
# Description:  Class NashFlow_spillback maintains list of FlowInterval instances; coordinates computation of dynamic equilibrium
# ===========================================================================

from collections import OrderedDict
from shutil import rmtree
import os
import time
from nashFlowClass import NashFlow
from flowIntervalClass_spillback import FlowInterval_spillback
from utilitiesClass import Utilities

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
        resettingEdges = [(v, w) for v, w in self.network.edges() if
                          Utilities.is_greater_tol(self.node_label(w, lowerBoundTime), self.node_label(v, lowerBoundTime) + self.network[v][w][
                              'transitTime'], TOL)] if lowerBoundTime > 0 else []
        fullEdges = [(v, w) for v, w in self.network.edges() if
                          self.is_full(v,w,self.node_label(v, lowerBoundTime))] if lowerBoundTime > 0 else []

        self.minInflowBound = min([self.network[v][w]['inflowBound'][self.node_label(v, lowerBoundTime)] for v,w in self.network.edges()]) if \
                                lowerBoundTime > 0 else min([self.network[v][w]['inCapacity'] for v, w in self.network.edges()])

        interval = FlowInterval_spillback(self.network, resettingEdges=resettingEdges, fullEdges=fullEdges, lowerBoundTime=lowerBoundTime,
                                inflowRate=self.inflowRate, minCapacity=self.minOutCapacity, counter=self.counter,
                                outputDirectory=self.rootPath, templateFile=self.templateFile, scipFile=self.scipFile,
                                timeout=self.timeout, minInflowBound=self.minInflowBound)

        if lowerBoundTime == 0:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(
                self.network)  # Compute shortest path network
        else:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(self.network, labels={
                v: self.node_label(v, lowerBoundTime) for v in self.network})  # Compute shortest path network

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

        interval.compute_alpha({node: self.node_label(node, lowerBoundTime) for node in self.network}, self.outflowBoundInformationDict)
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
            self.network[v][w]['queueSize'][vTimeLower] = 0

            self.network[v][w]['load'] = OrderedDict()
            self.network[v][w]['load'][0] = 0
            self.network[v][w]['load'][vTimeLower] = 0

            self.network[v][w]['inflowBound'] = OrderedDict()
            self.network[v][w]['inflowBound'][0] = self.network[v][w]['inCapacity']
            self.network[v][w]['inflowBound'][vTimeLower] = self.network[v][w]['inCapacity']

    def update_edge_properties(self, lowerBoundTime, interval):
        """
        Updates the edge properties after computation of a new flowInterval
        :param lowerBoundTime: lowerBoundTime of flowInterval
        :param interval: flowInterval
        """

        for v, w in self.network.edges():

            # Inflow changes
            vTimeLower, vTimeUpper = self.node_label(v, interval.lowerBoundTime), self.node_label(v, interval.upperBoundTime)
            inflowChangeBool = Utilities.is_not_eq_tol(interval.NTFNodeLabelDict[v],0)  # Can we extend the inflow interval?
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
                lastQueueSizeTime = next(reversed(self.network[v][w]['queueSize']))
                lastQueueSize = self.network[v][w]['queueSize'][lastQueueSizeTime]
                self.network[v][w]['queueSize'][vTimeUpper] = max(0, lastQueueSize + (
                        inflowVal - self.network[v][w]['outCapacity']) * (vTimeUpper - vTimeLower))

            self.animationIntervals[(v, w)].append(((vTimeLower, vTimeUpper), (wTimeLower, wTimeUpper)))


    def get_cumulative_inflow(self, v, w, t):
        """
        :param v: tail of edge
        :param w: head of edge
        :param t: time
        :return: F_(v,w)^+(t)
        """
        if Utilities.is_eq_tol(t,0):
            return 0
        for timeInterval, inflowVal in self.network[v][w]['inflow'].items():
            vTimeLower, vTimeUpper = timeInterval
            if vTimeLower <= time <= vTimeUpper:
                # This is the interval in which t lies
                return self.network[v][w]['cumulativeInflow'][vTimeLower] + inflowVal*(t-vTimeLower)

    def get_cumulative_outflow(self, v, w, t):
        """
        :param v: tail of edge
        :param w: head of edge
        :param t: time
        :return: F_(v,w)^-(t)
        """
        if Utilities.is_eq_tol(t,0):
            return 0
        for timeInterval, outflowVal in self.network[v][w]['outflow'].items():
            wTimeLower, wTimeUpper = timeInterval
            if wTimeLower <= time <= wTimeUpper:
                # This is the interval in which t lies
                return self.network[v][w]['cumulativeOutflow'][wTimeLower] + outflowVal*(t-wTimeLower)

    def arc_load(self, v, w ,t):
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
        return Utilities.is_eq_tol(self.arc_load(v,w,t), self.network[v][w]['storage'])
