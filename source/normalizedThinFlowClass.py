# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         normalizedThinFlowClass.py
# Description:  
# ===========================================================================
import networkx as nx
from utilitiesClass import Utilities
import os
from shutil import copy
import subprocess
import re


class NormalizedThinFlow:
    """description of class"""

    def __init__(self, shortestPathNetwork, id, resettingEdges, flowEdges, inflowRate, minCapacity, outputDirectory, templateFile, scipFile):

        self.network = shortestPathNetwork
        self.id = id
        self.resettingEdges = resettingEdges
        self.flowEdges = flowEdges
        self.inflowRate = inflowRate
        self.minCapacity = minCapacity
        self.outputDirectory = outputDirectory
        self.templateFile = templateFile
        self.scipFile = scipFile
        self.isValid = False

        self.run_order()

    def is_valid(self):
        return self.isValid

    def run_order(self):
        self.rootPath = os.path.join(self.outputDirectory, str(self.id) + '-NTF-' + Utilities.get_time())
        Utilities.create_dir(self.rootPath)

        copy(self.templateFile, self.rootPath)
        self.templateFile = os.path.join(self.rootPath, os.path.basename(self.templateFile))
        self.logFile = os.path.join(self.rootPath, 'outputLog.txt')

        self.write_zimpl_files()
        self.start_process()

        self.check_result()

    def get_labels_and_flow(self):
        labelPattern = r'l\$([\w]+)\s*([\d.]+)'

        flowPattern = r'x\$([\w]+)\$([\w]+)\s*([\d.]+)'

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
        cmd = [self.scipFile, '-f', self.templateFile, '-l', self.logFile]
        devNull = open(os.devnull, 'w')
        rc = subprocess.call(cmd, stdout=devNull)
        devNull.close()



    def write_zimpl_files(self):
        # Write files
        with open(os.path.join(self.rootPath, 'nodes.txt'), 'w') as nodeWriter:
            for node in self.network:
                nodeWriter.write(node + '\n')

        with open(os.path.join(self.rootPath, 'edges.txt'), 'w') as edgeWriter:
            for edge in self.network.edges():
                v, w = edge[0], edge[1]
                capacity = str(self.network[v][w]['capacity'])
                isInE_0 = str(1) if edge in self.flowEdges else str(0)
                isInE_Star = str(1) if edge in self.resettingEdges else str(0)
                outputString = v + ":" + w + ":" + capacity + ":" + isInE_0 + ":" + isInE_Star + '\n'
                edgeWriter.write(outputString)

        with open(os.path.join(self.rootPath, 'other.txt'), 'w') as otherWriter:
            otherWriter.write(str(self.inflowRate) + '\n')
            otherWriter.write(str(self.inflowRate/self.minCapacity) + '\n')






