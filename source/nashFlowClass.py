# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         nashFlowClass.py
# Description:  Class NashFlow maintains list of FlowInterval instances; coordinates computation of dynamic equilibrium
# ===========================================================================

from collections import OrderedDict
from shutil import rmtree
import os
import time
from flowIntervalClass import FlowInterval
from utilitiesClass import Utilities

# ======================================================================================================================

TOL = 1e-8  # Tolerance


class NashFlow:
    """Maintains Nash Flow over time"""

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
                                         'algorithm_' + str(templateFile + 1) + '.zpl')
        self.allInOne = (templateFile == 1)
        self.advancedAlgo = (templateFile == 2)  # If true, then advanced backtracking with preprocessing

        self.scipFile = scipFile
        self.cleanUpBool = cleanUpBool
        self.numberOfSolvedIPs = 0
        self.computationalTime = 0
        self.infinityReached = False  # True if last interval has alpha = +inf
        self.timeout = timeout

        self.minCapacity = Utilities.compute_min_capacity(self.network)
        self.counter = 0
        self.preprocessedNodes = 0
        self.preprocessedEdges = 0

        # Create directory for Nash-Flow
        self.rootPath = os.path.join(self.outputDirectory, 'NashFlow-' + Utilities.get_time())
        Utilities.create_dir(self.rootPath)

        self.flowIntervals = [] # List containing triplets of form (lowerBound, upperBound, FlowInterval-instance)
        self.lowerBoundsToIntervalDict = OrderedDict()
        self.animationIntervals = {edge: [] for edge in self.network.edges()}

    def run(self, nextIntervalOnly=False):
        """
        Compute the flow intervals up to self.numberOfIntervals
        :param nextIntervalOnly: If True, only the next flow interval is computed
        """
        computedUpperBound = 0
        k = 1 if self.numberOfIntervals != -1 else -float('inf')

        if nextIntervalOnly:
            Utilities.create_dir(self.rootPath)
            self.compute_flowInterval()
        else:
            while computedUpperBound < float('inf') and k <= self.numberOfIntervals:
                self.compute_flowInterval()
                computedUpperBound = self.flowIntervals[-1][1]
                k += 1

        # Clean up
        if self.cleanUpBool:
            rmtree(self.rootPath)

        div = float(len(self.flowIntervals))
        if not self.allInOne:
            totalBinaryList = [no for i in range(len(self.flowIntervals)) for no in self.flowIntervals[i][2].binaryVariableNumberList]  # To be deleted before handing in
        else:
            totalBinaryList = [(2*self.flowIntervals[i][2].shortestPathNetwork.number_of_edges() - len(self.flowIntervals[i][2].resettingEdges)) for i in range(len(self.flowIntervals)) for no in
                               self.flowIntervals[i][2].binaryVariableNumberList]

    def compute_flowInterval(self):
        """Method to compute a single flowInterval"""
        # Get lowerBoundTime
        lowerBoundTime = 0 if not self.flowIntervals else self.flowIntervals[-1][1]

        # Compute resettingEdges
        resettingEdges = [(v, w) for v, w in self.network.edges_iter() if
                          self.node_label(w, lowerBoundTime) > self.node_label(v, lowerBoundTime) + self.network[v][w][
                              'transitTime'] + TOL] if lowerBoundTime > 0 else []

        interval = FlowInterval(self.network, resettingEdges=resettingEdges, lowerBoundTime=lowerBoundTime,
                                inflowRate=self.inflowRate, minCapacity=self.minCapacity, counter=self.counter,
                                outputDirectory=self.rootPath, templateFile=self.templateFile, scipFile=self.scipFile,
                                timeout=self.timeout, )

        if lowerBoundTime == 0:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(
                self.network)  # Compute shortest path network
        else:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(self.network, labels={
                v: self.node_label(v, lowerBoundTime) for v in self.network})  # Compute shortest path network

        start = time.time()
        if self.advancedAlgo:
            interval.get_ntf_advanced()
        else:
            interval.get_ntf()

        end = time.time()
        self.computationalTime += (end - start)
        interval.computationalTime = end - start
        self.preprocessedNodes += interval.preprocessedNodes
        self.preprocessedEdges += interval.preprocessedEdges
        self.lowerBoundsToIntervalDict[lowerBoundTime] = interval

        interval.compute_alpha({node: self.node_label(node, lowerBoundTime) for node in self.network})
        self.flowIntervals.append((interval.lowerBoundTime, interval.upperBoundTime, interval))

        # Update in/out-flow rates
        if lowerBoundTime == 0:
            # init in/outflow
            for v, w in self.network.edges_iter():
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

        for v, w in self.network.edges_iter():

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
                lastQueueSizeTime = next(reversed(self.network[v][w]['queueSize']))
                lastQueueSize = self.network[v][w]['queueSize'][lastQueueSizeTime]
                self.network[v][w]['queueSize'][vTimeUpper] = max(0, lastQueueSize + (
                    inflowVal - self.network[v][w]['capacity']) * (vTimeUpper - vTimeLower))

            self.animationIntervals[(v, w)].append(((vTimeLower, vTimeUpper), (wTimeLower, wTimeUpper)))

        self.counter += 1
        self.numberOfSolvedIPs += interval.numberOfSolvedIPs

        self.infinityReached = (interval.alpha == float('inf'))

    def node_label(self, v, t):
        """
        Equivalent to l_v(time)
        :param v: node
        :param t: float
        :return: l_v(time)
        """
        if t == float('inf'):
            return float('inf')
        intervalLowerBoundTime = self.time_interval_correspondence(t)
        interval = self.lowerBoundsToIntervalDict[intervalLowerBoundTime]
        label = interval.shortestPathNetwork.node[v]['dist'] + (t - intervalLowerBoundTime) * \
                                                               interval.NTFNodeLabelDict[v]
        return label

    def queue_size(self, v, w, t):
        """
        Returns the queue size of edge e = vw given a timepoint t
        :param v: tail
        :param w: head
        :param t: timepoint
        :return: z_{vw}(t)
        """
        return self.cumulative_inflow(v, w, t) - self.cumulative_outflow(v, w,
                                                                         t + self.network[v][w]['transitTime'])

    def queue_delay(self, v, w, t):
        """
        Returns the queue delay of edge e = vw given a timepoint time
        :param v: tail
        :param w: head
        :param t: timepoint
        :return: q_{vw}(time)
        """
        return self.queue_size(v, w, t) / float(self.network[v][w]['capacity'])

    def cumulative_inflow(self, v, w, t):
        """
        Returns cumulative inflow of edge e=vw given time
        :param v: tail
        :param w: head
        :param t: timepoint
        :return: F^+_e(time)
        """
        if t <= TOL:
            return 0

        lastIntervalKey, lastIntervalValue = self.network[v][w]['inflow'].popitem(last=True)
        self.network[v][w]['inflow'][lastIntervalKey] = lastIntervalValue

        assert (Utilities.is_geq_tol(lastIntervalKey[1], t))

        integral = 0
        for lowerBound, upperBound in self.network[v][w]['inflow']:
            intervalInflow = self.network[v][w]['inflow'][(lowerBound, upperBound)]
            if t > upperBound + TOL:
                integral += (upperBound - lowerBound) * intervalInflow
            elif Utilities.is_geq_tol(t, lowerBound) and Utilities.is_geq_tol(upperBound, t):
                integral += (t - lowerBound) * intervalInflow
            else:
                break
        return integral

    def cumulative_outflow(self, v, w, t):
        """
        Returns cumulative outflow of edge e=vw given time
        :param v: tail
        :param w: head
        :param t: timepoint
        :return: F^-_e(time)
        """
        if t <= TOL:
            return 0

        lastIntervalKey, lastIntervalValue = self.network[v][w]['outflow'].popitem(last=True)
        self.network[v][w]['outflow'][lastIntervalKey] = lastIntervalValue

        assert (Utilities.is_geq_tol(lastIntervalKey[1], t))

        integral = 0
        for lowerBound, upperBound in self.network[v][w]['outflow']:
            intervalOutflow = self.network[v][w]['outflow'][(lowerBound, upperBound)]
            if t > upperBound + TOL:
                integral += (upperBound - lowerBound) * intervalOutflow
            elif Utilities.is_geq_tol(t, lowerBound) and Utilities.is_geq_tol(upperBound, t):
                integral += (t - lowerBound) * intervalOutflow
            else:
                break
        return integral

    def time_interval_correspondence(self, t):
        """
        :param t: timepoint
        :return: lastTime: lowerBound of flowInterval containing time
        """
        if Utilities.is_eq_tol(t, 0):
            return 0
        for lowerBoundTime in self.lowerBoundsToIntervalDict:
            if Utilities.is_geq_tol(t, lowerBoundTime):
                lastTime = lowerBoundTime
        return lastTime

    def get_stat_preprocessing(self):
        """Returns strings for preprocessing statistics"""
        if len(self.flowIntervals) == 0:
            return "N/A", "N/A"

        totalNodes, totalEdges = self.preprocessedNodes, self.preprocessedEdges
        avgNodes, avgEdges = float(totalNodes) / len(self.flowIntervals), float(totalEdges) / len(self.flowIntervals)
        return "%.2f" % avgNodes, "%.2f" % avgEdges

    def get_stat_solved_IPs(self):
        """Returns strings for No. of solved IPs statistics"""
        total = self.numberOfSolvedIPs
        if len(self.flowIntervals) == 0:
            return "N/A", "N/A"
        avg = float(total) / len(self.flowIntervals)
        return "%.2f" % avg, total

    def get_stat_time(self):
        """Returns strings for elapsed-time statistics"""
        total = self.computationalTime
        if len(self.flowIntervals) == 0:
            return "N/A", "N/A"
        avg = float(total) / len(self.flowIntervals)
        return "%.2f" % avg, "%.2f" % total
