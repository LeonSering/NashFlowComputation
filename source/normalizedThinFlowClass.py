# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         normalizedThinFlowClass.py
# Description:  
# ===========================================================================
import networkx as nx
from utilitiesClass import Utilities
import os
from itertools import combinations
from shutil import copy
import subprocess
import re

PATH_TO_SCIP    = '/home/doc/Documents/Thesis/dev/solver/scipoptsuite-4.0.0/scip-4.0.0/bin/scip'
TEMPLATE_DIR    = '/home/doc/Documents/Thesis/dev/NashFlowComputation/source/templates/V2.1/BIGM'
TEMPLATE_FILE   = '/home/doc/Documents/Thesis/dev/NashFlowComputation/source/templates/V2.1/BIGM/zimplTemplateV2_1_BIGM.zpl'

class NormalizedThinFlow:
    """description of class"""

    def __init__(self, shortestPathNetwork, id, resettingEdges, flowEdges, inflowRate, minCapacity, rootPath):

        self.network = shortestPathNetwork
        self.id = id
        self.resettingEdges = resettingEdges
        self.flowEdges = flowEdges
        self.inflowRate = inflowRate
        self.minCapacity = minCapacity
        self.rootPath = rootPath
        self.isValid = False

        self.run_order()

    def is_valid(self):
        return self.isValid

    def run_order(self):
        self.outputDirectory = os.path.join(self.rootPath, str(self.id) + '-NTF-' + Utilities.get_time())
        Utilities.create_dir(self.outputDirectory)

        copy(TEMPLATE_FILE, self.outputDirectory)
        self.templateFile = os.path.join(self.outputDirectory, os.path.basename(TEMPLATE_FILE))
        self.logFile = os.path.join(self.outputDirectory, 'outputLog.txt')

        self.write_zimpl_files()
        self.start_process()

        self.check_result()

    def get_labels_and_flow(self):
        labelPattern = r'l\$(\w)\s*(\d+)'
        flowPattern = r'x\$(\w)\$(\w)\s*(\d+)'

        labelMatch = re.findall(labelPattern, self.resultLog)
        labelDict = {node:float(val) for node, val in labelMatch}

        flowMatch = re.findall(flowPattern, self.resultLog)
        flowDict = {(v,w):float(val) for v, w, val in flowMatch}

        return labelDict, flowDict


    def check_result(self):
        pattern = r'[optimal solution found]' # Maybe switch to regex

        with open(self.logFile, 'r') as logFileReader:
            self.resultLog = logFileReader.read()
        self.isValid = ( pattern in self.resultLog )

    def start_process(self):
        cmd = [PATH_TO_SCIP, '-f', self.templateFile, '-l', self.logFile]
        devNull = open(os.devnull, 'w')
        rc = subprocess.call(cmd, stdout=devNull)
        devNull.close()



    def write_zimpl_files(self):
        # Write files
        with open(os.path.join(self.outputDirectory, 'nodes.txt'), 'w') as nodeWriter:
            for node in self.network:
                nodeWriter.write(node + '\n')

        with open(os.path.join(self.outputDirectory, 'edges.txt'), 'w') as edgeWriter:
            for edge in self.network.edges():
                v, w = edge[0], edge[1]
                capacity = str(self.network[v][w]['capacity'])
                isInE_0 = str(1) if edge in self.flowEdges else str(0)
                isInE_Star = str(1) if edge in self.resettingEdges else str(0)
                outputString = v + ":" + w + ":" + capacity + ":" + isInE_0 + ":" + isInE_Star + '\n'
                edgeWriter.write(outputString)

        with open(os.path.join(self.outputDirectory, 'other.txt'), 'w') as otherWriter:
            otherWriter.write(str(self.inflowRate) + '\n')
            otherWriter.write(str(self.minCapacity) + '\n')






