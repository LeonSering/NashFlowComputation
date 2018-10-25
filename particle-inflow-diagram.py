#=========================================================================
# Autho        Leon Sering
# Project:     NashFlowComputation Inflow-Time-Diagram
# File:        diagram.py
# Description: Skript to create a inflo-time-diagram
#=========================================================================
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import numpy as np
from source.nashFlowClass import NashFlow
from source.flowIntervalClass import FlowInterval
import time


# PARAMETERS

graph_path = "examples/laura_9.cg" # graph saved as pickle

u_lowerbound = 0
u_upperbound = 15 # upper bound on inflow rate
u_step = 0.1 # step size of inflow rate (going from 0 to u_bound (incl.))


#scipPath = "/net/site-local.linux64/bin/scip" # path to scip binaray
scipPath = "/usr/bin/scip" # path to scip binaray
outputDir = "output" # path for temporary files. (folder must exists before starting the script.)

timeLimit = 150 # only for the diagram, nash flows are always computed to the end

TOL = 1e-8 

figureUpdate = 10 # Number of Nash flow computation after which the figure.pdf is updated



# print plot to pdf function
fig, ax = plt.subplots()

currenttime = time.strftime("%Y%m%d-%H%M%S")

def printPdf(done = False):
    print clusterSplit

    # going from time to particle:    
    clusterJoinP = [[(a[0][0]*a[0][1], a[0][1]),(a[1][0]*a[1][1], a[1][1])] for a in clusterJoin]
    clusterSplitP = [[(a[0][0]*a[0][1], a[0][1]),(a[1][0]*a[1][1], a[1][1])] for a in clusterSplit] 
 
    lcJoin = mc.LineCollection(clusterJoinP, color = 'green', linewidths=0.2)
    ax.add_collection(lcJoin)
    lcSplit = mc.LineCollection(clusterSplitP, color = 'red', linewidths=0.2)
    ax.add_collection(lcSplit)
    
    
    for v, w in network.edges():
        e = (v, w)
        
        # going from time to particle:
        eventActiveParticle = [(a*b, b) for a, b in eventActive[e]]
        eventQueueParticle = [(a*b, b) for a, b in eventQueue[e]]
        
        plt.plot(*zip(*eventActiveParticle), label=str(v)+str(w)+'a')
        plt.plot(*zip(*eventQueueParticle), label=str(v)+str(w)+'q')

 #   ax.grid()
    legend = ax.legend()
#    plt.xlabel(r'time $\theta$')
    plt.xlabel(r'particle $\theta$')
    plt.ylabel(r'inflow $u_0$')
    ax.set_xlim(0, timeLimit)
    ax.set_ylim(u_lowerbound, u_upperbound)

    plt.savefig(outputDir + "/diagram_" + currenttime + ".pdf", format = "pdf")
    
    if done == True:
        plt.show()
    else: 
        ax.clear()


# loading network
print "Loading network: " + str(graph_path)
with open(graph_path, 'rb') as f:
    network = pickle.load(f)


counter = 0
eventActive = {}
eventQueue = {}
clusterJoin = []
clusterSplit = []
for v, w in network.edges():
    e = (v, w)
    eventActive[e] = []
    eventQueue[e] = []

oldInterval = {}
for inflowRate in np.arange(u_lowerbound, u_upperbound+u_step, u_step):
    counter += 1
    #computing nash flow
    print "Computing Nash flow with inflow rate: " + str(inflowRate)

    nashFlow = NashFlow(network, inflowRate, -1, outputDir, 0, scipPath, True, 0)
    nashFlow.run()
    print "Computation complete in " + "%.2f" % nashFlow.computationalTime + " seconds"

    intervalCounter = 0

    # creating data points
    for interval in nashFlow.flowIntervals:
        startTime = interval[0]
        endTime = interval[1]
        thinFlow = interval[2]
        if intervalCounter in oldInterval.keys():
            oldNodeLabel = oldInterval[intervalCounter][2].NTFNodeLabelDict
        for v, w in network.edges():
            e = (v, w)
            if e not in thinFlow.shortestPathNetwork.edges():
                if network[v][w]['transitTime'] + nashFlow.node_label(v, startTime) - nashFlow.node_label(w, startTime)  - thinFlow.alpha * (thinFlow.NTFNodeLabelDict[w] - thinFlow.NTFNodeLabelDict[v]) < TOL:
                    eventActive[e].append((endTime, inflowRate))
            elif e in thinFlow.resettingEdges:
                if  nashFlow.node_label(w, startTime)  - nashFlow.node_label(v, startTime)  - network[v][w]['transitTime'] + thinFlow.alpha * (thinFlow.NTFNodeLabelDict[w] - thinFlow.NTFNodeLabelDict[v]) < TOL:
                    eventQueue[e].append((endTime, inflowRate))
            if intervalCounter in oldInterval.keys():
                if e not in thinFlow.resettingEdges:
                    if abs(thinFlow.NTFNodeLabelDict[v] - thinFlow.NTFNodeLabelDict[w]) < TOL and abs(oldNodeLabel[v] - oldNodeLabel[w]) > TOL:
                        clusterJoin.append([(startTime, inflowRate - 0.5 * u_step), (endTime, inflowRate - 0.5 * u_step)])
                        print "JOIN inflowrat: " + str(inflowRate) + ", interval: " + str(intervalCounter) + ", arc: " + str(e) + ", l" +str(v) + " = " + str(thinFlow.NTFNodeLabelDict[v]) + ", l" +str(w) + " = " + str(thinFlow.NTFNodeLabelDict[w])  + ", oldl" +str(v) + " = " + str(oldNodeLabel[v]) + ", oldl" +str(w) + " = " + str(oldNodeLabel[w])
                    if abs(thinFlow.NTFNodeLabelDict[v] - thinFlow.NTFNodeLabelDict[w]) > TOL and abs(oldNodeLabel[v] - oldNodeLabel[w]) < TOL:
                        clusterSplit.append([(oldInterval[intervalCounter][0], inflowRate - u_step), (timeLimit*1000 if oldInterval[intervalCounter][1] == float('inf') else oldInterval[intervalCounter][1] , inflowRate - u_step)]) # Insert Split line to the old inflow rate for nice pictures
                        print "SPLIT inflowrat: " + str(inflowRate) + ", interval: " + str(intervalCounter) + ", arc: " + str(e) + ", l" +str(v) + " = " + str(thinFlow.NTFNodeLabelDict[v]) + ", l" +str(w) + " = " + str(thinFlow.NTFNodeLabelDict[w])  + ", oldl" +str(v) + " = " + str(oldNodeLabel[v]) + ", oldl" +str(w) + " = " + str(oldNodeLabel[w])

        oldInterval[intervalCounter] = interval
        intervalCounter += 1
        
    for k in oldInterval.keys(): # if there are less intervals then before the non existing intervals must be deleted
        if k >= intervalCounter:
            del oldInterval[k]
            print "delete interveral " + str(k)
    if counter % 10 == 0:
        printPdf()
print eventActive
print eventQueue

printPdf(True)

