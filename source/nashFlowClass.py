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

    def __init__(self, interface, graph, inflowRate, numberOfIntervals, outputDirectory, templateFile, scipFile):
        self.interface = interface
        self.network = graph.copy()
        self.inflowRate = inflowRate  # For the moment: constant
        self.numberOfIntervals = numberOfIntervals
        self.outputDirectory = outputDirectory
        self.templateFile = templateFile
        self.scipFile = scipFile

        for v, w in self.network.edges_iter():
            self.network[v][w]['inflow'] = OrderedDict()
            self.network[v][w]['outflow'] = OrderedDict()

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
            self.interface.add_last_interval_to_list()
            k += 1

    def compute_flowInterval(self):
        # NOTE TO MYSELF: computing shortest paths and resetting edges is only necessary for first flowInterval -> later: implement in flowIntervallClass

        #get lowerBoundTime
        lowerBoundTime = 0 if not self.flowIntervals else self.flowIntervals[-1][1]

        #compute resettingEdges
        # method using self.queue_size might lead to problems, as outflow could not be defined properly
        #resettingEdges = [(v,w) for v, w in self.network.edges_iter() if self.queue_size(v,w,self.node_label(v,lowerBoundTime)) > TOL] if lowerBoundTime > 0 else []
        resettingEdges = [(v,w) for v, w in self.network.edges_iter() if self.node_label(w, lowerBoundTime) > self.node_label(v, lowerBoundTime) + self.network[v][w]['transitTime'] + TOL] if lowerBoundTime > 0 else []

        interval = FlowInterval(self.network, resettingEdges=resettingEdges, lowerBoundTime=lowerBoundTime, inflowRate=self.inflowRate, minCapacity=self.minCapacity, counter=self.counter, outputDirectory=self.rootPath, templateFile=self.templateFile, scipFile=self.scipFile)

        if lowerBoundTime == 0:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(self.network, lowerBoundTime)  # Compute shortest path network
        else:
            interval.shortestPathNetwork = Utilities.get_shortest_path_network(self.network, lowerBoundTime, labels={v:self.node_label(v, lowerBoundTime) for v in self.network})  # Compute shortest path network

        interval.get_NTF()
        self.lowerBoundsToIntervalDict[lowerBoundTime] = interval

        interval.compute_alpha({node:self.node_label(node, lowerBoundTime) for node in self.network})
        self.flowIntervals.append((interval.lowerBoundTime, interval.upperBoundTime, interval))

        # Update in-flow rates
        if lowerBoundTime == 0:
            #init inflow
            for v, w in self.network.edges_iter():
                self.network[v][w]['inflow'][(0, self.node_label(v, 0))] = 0
                self.network[v][w]['outflow'][(0, self.node_label(w, 0))] = 0

        for v, w in self.network.edges_iter():
            if Utilities.is_not_eq_tol(interval.NTFNodeLabelDict[v], 0):
                self.network[v][w]['inflow'][(self.node_label(v, interval.lowerBoundTime), self.node_label(v, interval.upperBoundTime))] = interval.NTFEdgeFlowDict[(v,w)]/interval.NTFNodeLabelDict[v]
            if Utilities.is_not_eq_tol(interval.NTFNodeLabelDict[w], 0):
                self.network[v][w]['outflow'][(self.node_label(w, interval.lowerBoundTime), self.node_label(w, interval.upperBoundTime))] = interval.NTFEdgeFlowDict[(v,w)]/interval.NTFNodeLabelDict[w]

        self.counter += 1

    def node_label(self, v, time):
        intervalLowerBoundTime = self.time_interval_correspondence(time)
        interval = self.lowerBoundsToIntervalDict[intervalLowerBoundTime]
        label = interval.shortestPathNetwork.node[v]['dist'] + (time-intervalLowerBoundTime)*interval.NTFNodeLabelDict[v]
        return label

    def queue_size(self, v, w, time):
        return self.cumulative_inflow(v, w, time) - self.cumulative_outflow(v, w, time + self.network[v][w]['transitTime'])

    def queue_delay(self, v, w, time):
        return self.queue_size(v, w, time)/self.network[v][w]['capacity']

    def cumulative_inflow(self, v, w, time):
        if time <= TOL:
            return 0

        lastIntervalKey, lastIntervalValue = self.network[v][w]['inflow'].popitem(last=True)
        self.network[v][w]['inflow'][lastIntervalKey] = lastIntervalValue

        #assert( time <= lastIntervalKey[1] )
        assert( Utilities.is_geq_tol(lastIntervalKey[1], time))

        integral = 0
        for lowerBound, upperBound in self.network[v][w]['inflow']:
            if time > upperBound + TOL:
                integral += (upperBound - lowerBound) * self.network[v][w]['inflow'][(lowerBound, upperBound)]
            elif Utilities.is_geq_tol(time, lowerBound) and Utilities.is_geq_tol(upperBound, time):
                integral += (time - lowerBound) * self.network[v][w]['inflow'][(lowerBound, upperBound)]
            else:
                break
        return integral

    def cumulative_outflow(self, v, w, time):
        if time <= TOL:
            return 0

        lastIntervalKey, lastIntervalValue = self.network[v][w]['outflow'].popitem(last=True)
        self.network[v][w]['outflow'][lastIntervalKey] = lastIntervalValue

        #assert( time <= lastIntervalKey[1] )
        assert (Utilities.is_geq_tol(lastIntervalKey[1], time))

        integral = 0
        for lowerBound, upperBound in self.network[v][w]['outflow']:
            if time > upperBound + TOL:
                integral += (upperBound - lowerBound) * self.network[v][w]['outflow'][(lowerBound, upperBound)]
            elif Utilities.is_geq_tol(time, lowerBound) and Utilities.is_geq_tol(upperBound, time):
                integral += (time - lowerBound) * self.network[v][w]['outflow'][(lowerBound, upperBound)]
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


#Instance for debugging
'''
#TRIVIAL, first alpha = inf
inflowRate = 10
nodes = ['s', 'a', 'b', 't']
edges = [('s', 'a'),  ('a', 't'), ('a', 'b'), ('b', 't')]
transitTimeList = [1, 2, 1, 4]
transitTimeDict = {entry[0]:entry[1] for entry in zip(edges, transitTimeList)}
capacityList = [5, 7, 2, 5]
capacityDict = {entry[0]:entry[1] for entry in zip(edges, capacityList)}

nf = NashFlow(nodes, edges, transitTimeDict, capacityDict, inflowRate)

inflowRate = 20
nodes = ['s', 'v', 'w', 't']
edges = [('s', 'v'),  ('s', 'w'), ('v', 'w'), ('v', 't'), ('w', 't')]
transitTimeList = [1, 6, 1, 5, 1]
transitTimeDict = {entry[0]:entry[1] for entry in zip(edges, transitTimeList)}
capacityList = [2, 1, 1, 2, 1]
capacityDict = {entry[0]:entry[1] for entry in zip(edges, capacityList)}

nf = NashFlow(nodes, edges, transitTimeDict, capacityDict, inflowRate)
'''