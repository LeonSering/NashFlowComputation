# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         nashFlowClass.py
# Description:  
# ===========================================================================
import networkx as nx
from collections import OrderedDict
from flowIntervalClass import FlowInterval
from utilitiesClass import Utilities
import os

TOL = 1e-8

class NashFlow:
    """description of class"""

    def __init__(self, graph, inflowRate, numberOfIntervals, outputDirectory, templateFile, scipFile, cleanUpBool):
        self.network = graph.copy()
        self.inflowRate = inflowRate  # For the moment: constant
        self.numberOfIntervals = numberOfIntervals
        self.outputDirectory = outputDirectory
        self.templateFile = templateFile
        self.scipFile = scipFile
        self.cleanUpBool = cleanUpBool
        self.numberOfSolvedIPs = 0
        self.infinityReached = False # True if last interval has alpha = +inf


        self.minCapacity = Utilities.compute_min_capacity(self.network)
        self.counter = 0

        # Create directory for Nash-Flow
        self.rootPath = os.path.join(self.outputDirectory, 'NashFlow-' + Utilities.get_time())
        Utilities.create_dir(self.rootPath)

        self.flowIntervals = []
        self.lowerBoundsToIntervalDict = OrderedDict()



    def run(self):
        computedUpperBound = 0
        k = 1 if self.numberOfIntervals != -1 else -float('inf')

        while computedUpperBound < float('inf') and k <= self.numberOfIntervals:
            self.compute_flowInterval()
            computedUpperBound = self.flowIntervals[-1][1]
            k += 1


    def compute_flowInterval(self):
        # NOTE TO MYSELF: computing shortest paths and resetting edges is only necessary for first flowInterval -> later: implement in flowIntervallClass

        #get lowerBoundTime
        lowerBoundTime = 0 if not self.flowIntervals else self.flowIntervals[-1][1]

        #compute resettingEdges
        # method using self.queue_size might lead to problems, as outflow could not be defined properly
        #resettingEdges = [(v,w) for v, w in self.network.edges_iter() if self.queue_size(v,w,self.node_label(v,lowerBoundTime)) > TOL] if lowerBoundTime > 0 else []
        resettingEdges = [(v,w) for v, w in self.network.edges_iter() if self.node_label(w, lowerBoundTime) > self.node_label(v, lowerBoundTime) + self.network[v][w]['transitTime'] + TOL] if lowerBoundTime > 0 else []

        interval = FlowInterval(self.network, resettingEdges=resettingEdges, lowerBoundTime=lowerBoundTime, inflowRate=self.inflowRate, minCapacity=self.minCapacity, counter=self.counter, outputDirectory=self.rootPath, templateFile=self.templateFile, scipFile=self.scipFile, cleanUpBool=self.cleanUpBool)

        if lowerBoundTime == 0:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(self.network, lowerBoundTime)  # Compute shortest path network
        else:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(self.network, lowerBoundTime, labels={v:self.node_label(v, lowerBoundTime) for v in self.network})  # Compute shortest path network

        interval.get_NTF()
        self.lowerBoundsToIntervalDict[lowerBoundTime] = interval

        interval.compute_alpha({node:self.node_label(node, lowerBoundTime) for node in self.network})
        self.flowIntervals.append((interval.lowerBoundTime, interval.upperBoundTime, interval))

        # Update in/out-flow rates
        if lowerBoundTime == 0:
            #init in/outflow
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
            vTimeLower, vTimeUpper = self.node_label(v, interval.lowerBoundTime), self.node_label(v, interval.upperBoundTime)
            inflowChangeBool = Utilities.is_not_eq_tol(interval.NTFNodeLabelDict[v], 0) # Can we extend the inflow interval?
            inflowVal = interval.NTFEdgeFlowDict[(v, w)] / interval.NTFNodeLabelDict[v] if inflowChangeBool else 0
            if inflowChangeBool:
                self.network[v][w]['inflow'][(vTimeLower, vTimeUpper)] = inflowVal

            if vTimeUpper < float('inf'):
                vLastTime = next(reversed(self.network[v][w]['cumulativeInflow']))
                self.network[v][w]['cumulativeInflow'][vTimeUpper] = self.network[v][w]['cumulativeInflow'][vLastTime] + inflowVal * (vTimeUpper-vTimeLower)

            # Outflow changes
            wTimeLower, wTimeUpper = self.node_label(w, interval.lowerBoundTime), self.node_label(w, interval.upperBoundTime)
            outflowChangeBool = Utilities.is_not_eq_tol(interval.NTFNodeLabelDict[w], 0) # Can we extend the outflow interval?
            outflowVal = interval.NTFEdgeFlowDict[(v, w)] / interval.NTFNodeLabelDict[w] if outflowChangeBool else 0
            if outflowChangeBool:
                self.network[v][w]['outflow'][(wTimeLower, wTimeUpper)] = outflowVal

            if wTimeUpper < float('inf'):
                wLastTime = next(reversed(self.network[v][w]['cumulativeOutflow']))
                self.network[v][w]['cumulativeOutflow'][wTimeUpper] = self.network[v][w]['cumulativeOutflow'][wLastTime] + outflowVal * (wTimeUpper-wTimeLower)

            # Queue size changes
            if vTimeUpper < float('inf'):
                lastQueueSizeTime = next(reversed(self.network[v][w]['queueSize']))
                lastQueueSize = self.network[v][w]['queueSize'][lastQueueSizeTime]
                self.network[v][w]['queueSize'][vTimeUpper] = max(0, lastQueueSize + (inflowVal - self.network[v][w]['capacity'])*(vTimeUpper-vTimeLower))



        self.counter += 1
        self.numberOfSolvedIPs += interval.numberOfSolvedIPs

        self.infinityReached = ( interval.alpha == float('inf') )


    def node_label(self, v, time):
        if time == float('inf'):
            return float('inf')
        intervalLowerBoundTime = self.time_interval_correspondence(time)
        interval = self.lowerBoundsToIntervalDict[intervalLowerBoundTime]
        label = interval.shortestPathNetwork.node[v]['dist'] + (time-intervalLowerBoundTime)*interval.NTFNodeLabelDict[v]
        return label

    def queue_size(self, v, w, time):
        return self.cumulative_inflow(v, w, time) - self.cumulative_outflow(v, w, time + self.network[v][w]['transitTime'])

    def queue_delay(self, v, w, time):
        return self.queue_size(v, w, time)/float(self.network[v][w]['capacity'])

    def cumulative_inflow(self, v, w, time):
        if time <= TOL:
            return 0

        lastIntervalKey, lastIntervalValue = self.network[v][w]['inflow'].popitem(last=True)
        self.network[v][w]['inflow'][lastIntervalKey] = lastIntervalValue

        assert( Utilities.is_geq_tol(lastIntervalKey[1], time))

        integral = 0
        for lowerBound, upperBound in self.network[v][w]['inflow']:
            intervalInflow = self.network[v][w]['inflow'][(lowerBound, upperBound)]
            if time > upperBound + TOL:
                integral += (upperBound - lowerBound) * intervalInflow
            elif Utilities.is_geq_tol(time, lowerBound) and Utilities.is_geq_tol(upperBound, time):
                integral += (time - lowerBound) * intervalInflow
            else:
                break
        return integral

    def cumulative_inflow_work_in_progress(self, v, w, time):
        if time <= TOL:
            return 0

        lastCumulativeInflowTime = next(reversed(self.network[v][w]['cumulativeInflow']))


        assert( Utilities.is_geq_tol(lastCumulativeInflowTime, time) or self.infinityReached)

        # Find position of element to the left
        timesList = self.network[v][w]['cumulativeInflow'].keys()
        pos = Utilities.get_insertion_point_left(timesList, time)
        if pos == 0:
            return 0
        elif pos > len(timesList):
            lastTime = timesList[pos-1]
            inflow = self.network[v][w]['inflow'][next(reversed(self.network[v][w]['inflow']))]
            return self.network[v][w]['cumulativeInflow'][lastTime] + (time-lastTime)*inflow
        else:
            pass


    def cumulative_outflow(self, v, w, time):
        if time <= TOL:
            return 0

        lastIntervalKey, lastIntervalValue = self.network[v][w]['outflow'].popitem(last=True)
        self.network[v][w]['outflow'][lastIntervalKey] = lastIntervalValue

        assert (Utilities.is_geq_tol(lastIntervalKey[1], time))

        integral = 0
        for lowerBound, upperBound in self.network[v][w]['outflow']:
            intervalOutflow = self.network[v][w]['outflow'][(lowerBound, upperBound)]
            if time > upperBound + TOL:
                integral += (upperBound - lowerBound) * intervalOutflow
            elif Utilities.is_geq_tol(time, lowerBound) and Utilities.is_geq_tol(upperBound, time):
                integral += (time - lowerBound) * intervalOutflow
            else:
                break
        return integral

    def time_interval_correspondence(self, time):
        if Utilities.is_eq_tol(time, 0):
            return 0
        for lowerBoundTime in self.lowerBoundsToIntervalDict:
            if Utilities.is_geq_tol(time, lowerBoundTime):
                lastTime = lowerBoundTime
        return lastTime


