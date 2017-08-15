# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         normalizedThinFlowClass.py
# Description:  Data structure for thin flow and computation methods
# ===========================================================================

import os
import re
import subprocess
from shutil import copy
from utilitiesClass import Utilities


# ======================================================================================================================

class NormalizedThinFlow:
    """Data structure for thin flow and computation methods"""

    def __init__(self, shortestPathNetwork, id, resettingEdges, flowEdges, inflowRate, minCapacity, outputDirectory,
                 templateFile, scipFile):
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

        self.network = shortestPathNetwork
        self.id = id
        self.resettingEdges = resettingEdges
        self.flowEdges = flowEdges  # = E_0
        self.inflowRate = inflowRate
        self.minCapacity = minCapacity
        self.outputDirectory = outputDirectory
        self.templateFile = templateFile
        self.scipFile = scipFile
        self.isValid = False  # True if thin flow w.r.t. E_0 has been found

    def is_valid(self):
        return self.isValid

    def run_order(self):
        """Execution order"""
        self.rootPath = os.path.join(self.outputDirectory, str(self.id) + '-NTF-' + Utilities.get_time())
        Utilities.create_dir(self.rootPath)

        copy(self.templateFile, self.rootPath)
        self.templateFile = os.path.join(self.rootPath, os.path.basename(self.templateFile))
        self.logFile = os.path.join(self.rootPath, 'outputLog.txt')

        self.write_zimpl_files()
        self.start_process()

        self.check_result()

    def get_labels_and_flow(self):
        """Extract labels and flow values from log file using regex"""
        labelPattern = r'l\$([\w]+)\s*([\d.e-]+)'

        flowPattern = r'x\$([\w]+)\$([\w]+)\s*([\d.e-]+)'

        labelMatch = re.findall(labelPattern, self.resultLog)
        labelDict = {node: float(val) for node, val in labelMatch}

        flowMatch = re.findall(flowPattern, self.resultLog)
        flowDict = {(v, w): float(val) for v, w, val in flowMatch}

        return labelDict, flowDict

    def check_result(self):
        """Check whether NTF exists"""
        pattern = r'[optimal solution found]'  # Maybe switch to regex

        with open(self.logFile, 'r') as logFileReader:
            self.resultLog = logFileReader.read()
        self.isValid = (pattern in self.resultLog)

    def start_process(self):
        """Start SCIP process"""
        cmd = [self.scipFile, '-f', self.templateFile, '-l', self.logFile]
        devNull = open(os.devnull, 'w')  # SCIP saves logs itself, no need for stdout of process
        self.proc = subprocess.Popen(cmd, stdout=devNull, stderr=devNull)
        self.proc.communicate()

    def write_zimpl_files(self):
        """Write the ZIMPL files"""
        with open(os.path.join(self.rootPath, 'nodes.txt'), 'w') as nodeWriter:
            for node in self.network:
                nodeWriter.write(str(node) + '\n')

        with open(os.path.join(self.rootPath, 'edges.txt'), 'w') as edgeWriter:
            for edge in self.network.edges():
                v, w = edge[0], edge[1]
                capacity = str(self.network[v][w]['capacity'])
                isInE_0 = str(1) if edge in self.flowEdges else str(0)
                isInE_Star = str(1) if edge in self.resettingEdges else str(0)
                outputString = str(v) + ":" + str(w) + ":" + capacity + ":" + isInE_0 + ":" + isInE_Star + '\n'
                edgeWriter.write(outputString)

        with open(os.path.join(self.rootPath, 'other.txt'), 'w') as otherWriter:
            otherWriter.write(str(self.inflowRate) + '\n')
            M = max(1, self.inflowRate / self.minCapacity)  # BIG M
            otherWriter.write(str(M) + '\n')
