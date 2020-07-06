# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         normalizedThinFlowClass_spillback.py
# Description:  Data structure for spillback thin flow and computation methods
# ===========================================================================

import os
import re

from source.normalizedThinFlowClass import NormalizedThinFlow


# ======================================================================================================================

class NormalizedThinFlow_spillback(NormalizedThinFlow):
    """Data structure for thin flow and computation methods"""

    def __init__(self, shortestPathNetwork, id, resettingEdges, flowEdges, inflowRate, minCapacity, outputDirectory,
                 templateFile, scipFile, minInflowBound):
        """
        :param shortestPathNetwork: Networkx DiGraph instance
        :param id: unique ID for directory creation
        :param resettingEdges: list of resetting edges
        :param flowEdges: list of edges that could send flow, i.e. E_0
        :param inflowRate: u_0
        :param minCapacity: minimal capacity of all edges
        :param outputDirectory: directory where logs are saved
        :param templateFile: path to template file
        :param scipFile: path to scip binary
        """

        NormalizedThinFlow.__init__(self, shortestPathNetwork, id, resettingEdges, flowEdges, inflowRate, minCapacity,
                                    outputDirectory,
                                    templateFile, scipFile)

        self.minInflowBound = minInflowBound

    def get_labels_and_flow(self):
        """Extract labels, spillback-factors and flow values from log file using regex"""
        labelPattern = r'l\$([\w]+)\s*([\d.e-]+)'
        spillbackFactorPattern = r'c\$([\w]+)\s*([\d.e-]+)'
        flowPattern = r'x\$([\w]+)\$([\w]+)\s*([\d.e-]+)'

        labelMatch = re.findall(labelPattern, self.resultLog)
        labelDict = {node: float(val) for node, val in labelMatch}

        spillbackFactorMatch = re.findall(spillbackFactorPattern, self.resultLog)
        spillbackFactorDict = {node: float(val) for node, val in spillbackFactorMatch}

        flowMatch = re.findall(flowPattern, self.resultLog)
        flowDict = {(v, w): float(val) for v, w, val in flowMatch}

        return labelDict, spillbackFactorDict, flowDict

    def write_zimpl_files(self):
        """Write the ZIMPL files"""
        with open(os.path.join(self.rootPath, 'nodes.txt'), 'w') as nodeWriter:
            for node in self.network:
                nodeWriter.write(str(node) + '\n')

        with open(os.path.join(self.rootPath, 'edges.txt'), 'w') as edgeWriter:
            for edge in self.network.edges():
                v, w = edge[0], edge[1]
                capacity = str(self.network[v][w]['outCapacity'])
                inflowBound = str(self.network[v][w]['inflowBound']) if self.network[v][w]['inflowBound'] != float('inf') else str(-1)
                isInE_0 = str(1) if edge in self.flowEdges else str(0)
                isInE_Star = str(1) if edge in self.resettingEdges else str(0)
                outputString = str(v) + ":" + str(
                    w) + ":" + capacity + ":" + inflowBound + ":" + isInE_0 + ":" + isInE_Star + '\n'
                edgeWriter.write(outputString)

        with open(os.path.join(self.rootPath, 'other.txt'), 'w') as otherWriter:
            otherWriter.write(str(self.inflowRate) + '\n')
            M = max(1, self.inflowRate / self.minCapacity, self.inflowRate / self.minInflowBound)  # BIG M
            otherWriter.write(str(M) + '\n')
