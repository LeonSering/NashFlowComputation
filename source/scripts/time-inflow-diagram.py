# ===========================================================================
# Author:       Leon SERING
# Project:      NashFlowComputation 2018
# File:         time-inflow-diagram.py
# Description:  Script to create Inflow-time-diagram
# ===========================================================================

import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

from source.nashFlowClass import NashFlow

# PARAMETERS
graph_path = "koch_skutella_example.cg"  # graph saved as pickle

u_bound = 10.0  # upper bound on inflow rate
u_step = 0.1  # step size of inflow rate (going from 0 to u_bound (incl.))

scipPath = "/net/site-local.linux64/bin/scip"  # path to scip binary
outputDir = "output"  # path for temporary files. (folder must exists before starting the script. diagram.pdf is always created on the same folder as the script file)

timeLimit = 15  # only for the diagram, nash flows are always computed to the end

TOL = 1e-8

figureUpdate = 10  # Number of Nash flow computation after which the figure.pdf is updated

# print plot to pdf function
fig, ax = plt.subplots()
currenttime = time.strftime("%Y%m%d-%H%M%S")


def printPdf(done=False):
    for v, w in network.edges():
        e = (v, w)
        plt.plot(*zip(*eventActive[e]), label=str(v) + str(w) + 'a')
        plt.plot(*zip(*eventQueue[e]), label=str(v) + str(w) + 'q')

    legend = ax.legend()

    ax.set_xlim(0, timeLimit)
    ax.set_ylim(0, u_bound)

    plt.savefig("diagram_" + currenttime + ".pdf", format="pdf")

    if done:
        plt.show()
    ax.clear()


# loading network
print "Loading network: " + str(graph_path)
with open(graph_path, 'rb') as f:
    network = pickle.load(f)

counter = 0
eventActive = {}
eventQueue = {}
for v, w in network.edges():
    e = (v, w)
    eventActive[e] = []
    eventQueue[e] = []

for inflowRate in np.arange(0.0, u_bound + u_step, u_step):
    counter += 1
    # computing nash flow
    print "Computing Nash flow with inflow rate: " + str(inflowRate)

    nashFlow = NashFlow(network, inflowRate, -1, outputDir, 0, scipPath, True, 0)
    nashFlow.run()
    print "Computation complete in " + "%.2f" % nashFlow.computationalTime + " seconds"

    # creating data points
    for interval in nashFlow.flowIntervals:
        startTime = interval[0]
        endTime = interval[1]
        thinFlow = interval[2]
        for v, w in network.edges():
            e = (v, w)
            if e not in thinFlow.shortestPathNetwork.edges():
                if network[v][w]['transitTime'] + nashFlow.node_label(v, startTime) - nashFlow.node_label(w,
                                                                                                          startTime) - thinFlow.alpha * (
                        thinFlow.NTFNodeLabelDict[w] - thinFlow.NTFNodeLabelDict[v]) < TOL:
                    eventActive[e].append((endTime, inflowRate))
            elif e in thinFlow.resettingEdges:
                if nashFlow.node_label(w, startTime) - nashFlow.node_label(v, startTime) - network[v][w][
                    'transitTime'] + thinFlow.alpha * (
                        thinFlow.NTFNodeLabelDict[w] - thinFlow.NTFNodeLabelDict[v]) < TOL:
                    eventQueue[e].append((endTime, inflowRate))
    if counter % 10 == 0:
        printPdf()
print eventActive
print eventQueue

printPdf(True)
