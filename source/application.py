# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         application.py
# Description:  Interface class; controlling signals/slots & communication between widgets
# ===========================================================================

import configparser
import os
import pickle
import subprocess
import sys
import threading
import time
import warnings
from tempfile import gettempdir

import networkx as nx
from PyQt5 import QtGui, QtCore, QtWidgets

from source.nashFlowClass import NashFlow
from source.nashFlowClass_spillback import NashFlow_spillback
from source.plotAnimationCanvasClass import PlotAnimationCanvas
from source.plotCanvasClass import PlotCanvas
from source.plotNTFCanvasClass import PlotNTFCanvas
from source.plotQueueCanvasClass import PlotQueueCanvas
from source.plotValuesCanvasClass import PlotValuesCanvas
from source.utilitiesClass import Utilities
import source.ui.mainWdw as mainWdw


# =======================================================================================================================
warnings.filterwarnings('ignore')  # Suppress GTK-Warnings
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)  # This is necessary if threads access the GUI

class Interface(QtWidgets.QMainWindow, mainWdw.Ui_MainWindow):
    """Controls GUI"""

    def __init__(self):
        """Initialization of Class and GUI"""
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        # Scaling factors of frames, to avoid distortion
        self.plotCanvasStretchFactor = float(self.plotFrame_general.width()) / self.plotFrame_general.height()
        self.plotAnimationCanvasStretchFactor = float(
            self.plotAnimationFrame_general.width()) / self.plotAnimationFrame_general.height()
        self.plotNTFCanvasStretchFactor = float(self.plotNTFFrame_general.width()) / self.plotNTFFrame_general.height()
        # self.plotQueueCanvasStretchFactor = float(self.plotQueueFrame_general.width()) / self.plotQueueFrame_general.height()

        self.tfTypeList = ['general', 'spillback']
        self.gttr('tabWidget', 'spillback').setCurrentIndex(
            0)  # Note: This has to be done first, otherwise self.currentTF is wrong
        self.tabWidget.setCurrentIndex(0)  # Show General Tab
        self.gttr('tabWidget', 'general').setCurrentIndex(0)  # Show Graph creation Tab
        self.currentTF = self.tfTypeList[self.tabWidget.currentIndex()]  # Currently selected tab information

        # Init graphs and other config defaults
        for tfType in self.tfTypeList:
            self.sttr('network', tfType, self.init_graph())
            self.gttr('network', tfType).graph['type'] = tfType
            self.sttr('animationLowerBound', tfType, 0)
            self.sttr('animationUpperBound', tfType, 1)
            self.sttr('animationRunning', tfType, False)
            self.gttr('inflowLineEdit', tfType).setText('1')  # Default value

        # Config defaults
        self.outputDirectory = ''
        self.templateFile = 0  # 0,1,2 for three algorithms from thesis (0 in case of spillback)
        self.scipFile = ''
        self.timeoutActivated = False
        self.defaultLoadSaveDir = ''
        self.numberOfIntervals = -1
        self.cleanUpEnabled = True

        self.configFile = configparser.ConfigParser()  # This is the parser, not to confuse with the actual config.txt File, which cannot be specified

        # Initializations
        self.init_app()
        self.load_config()  # Try to load configuration file

        # Signal configuration
        for tfType in self.tfTypeList:
            self.gttr('updateNodeButton', tfType).clicked.connect(self.update_node)
            self.gttr('deleteNodeButton', tfType).clicked.connect(self.delete_node)
            self.gttr('updateEdgeButton', tfType).clicked.connect(self.update_add_edge)
            self.gttr('deleteEdgeButton', tfType).clicked.connect(self.delete_edge)
            self.gttr('nodeSelectionListWidget', tfType).clicked.connect(self.update_focus_node)
            self.gttr('edgeSelectionListWidget', tfType).clicked.connect(self.update_focus_edge)
            self.gttr('computeFlowPushButton', tfType).clicked.connect(self.compute_nash_flow)
            self.gttr('exportDiagramPushButton', tfType).clicked.connect(self.export_diagram)
            self.gttr('setTimePushButton', tfType).clicked.connect(self.set_new_time_manually)
            self.gttr('showEdgesWithoutFlowCheckBox', tfType).clicked.connect(self.change_no_flow_show_state)
            self.gttr('playPushButton', tfType).clicked.connect(self.play_animation)
            self.gttr('pausePushButton', tfType).clicked.connect(self.pause_animation)
            self.gttr('stopPushButton', tfType).clicked.connect(self.stop_animation)
            self.gttr('recordPushButton', tfType).clicked.connect(self.export_animation)
            self.gttr('computeIntervalPushButton', tfType).clicked.connect(self.compute_next_interval)
            self.gttr('intervalsListWidget', tfType).clicked.connect(self.update_ntf_display)
            self.gttr('setTimeLineEdit', tfType).returnPressed.connect(self.set_new_time_manually)

            # Config signals
            self.gttr('outputDirectoryPushButton', tfType).clicked.connect(self.select_output_directory)
            self.gttr('scipPathPushButton', tfType).clicked.connect(self.select_scip_binary)
            self.gttr('cleanUpCheckBox', tfType).clicked.connect(self.change_cleanup_state)
            self.gttr('activateTimeoutCheckBox', tfType).clicked.connect(self.change_timeout_state)

            # Keyboard shortcuts
            self.gttr('transitTimeLineEdit', tfType).returnPressed.connect(self.update_add_edge)
            self.gttr('nodeNameLineEdit', tfType).returnPressed.connect(self.update_node)
            self.gttr('nodeXLineEdit', tfType).returnPressed.connect(self.update_node)
            self.gttr('nodeYLineEdit', tfType).returnPressed.connect(self.update_node)

            # Configure Slider
            self.gttr('timeSlider', tfType).setMinimum(0)
            self.gttr('timeSlider', tfType).setMaximum(99)
            self.gttr('timeSlider', tfType).setValue(0)
            # self.gttr('timeSlider', tfType).setTickPosition(2) # Set ticks below horizontal slider
            self.gttr('timeSlider', tfType).setTickInterval(1)

            # Slider signals
            self.gttr('timeSlider', tfType).valueChanged.connect(self.slider_value_change)
            self.gttr('timeSlider', tfType).sliderReleased.connect(self.slider_released)

        # General-only signals
        self.capacityLineEdit_general.returnPressed.connect(self.update_add_edge)

        # Spillback-only signals
        self.inCapacityLineEdit_spillback.returnPressed.connect(self.update_add_edge)
        self.outCapacityLineEdit_spillback.returnPressed.connect(self.update_add_edge)
        self.storageLineEdit_spillback.returnPressed.connect(self.update_add_edge)

        # Non-assigned Signals
        self.tabWidget.currentChanged.connect(self.tabSwitched)
        self.actionNew_graph.triggered.connect(self.re_init_app)
        self.actionLoad_graph.triggered.connect(self.load_graph)
        self.actionSave_graph.triggered.connect(self.save_graph)
        self.actionExit.triggered.connect(QtWidgets.QApplication.quit)
        #self.actionLoad_Nashflow.triggered.connect(self.load_nashflow)
        #self.actionSave_Nashflow.triggered.connect(self.save_nashflow)
        self.actionOpen_ThinFlowComputation.triggered.connect(self.open_tfc)
        self.actionMove_graph_to_ThinFlowComputation.triggered.connect(self.move_to_tfc)
        self.actionChange_animation_time_range.triggered.connect(self.generate_animation_dialog)
        self.actionOpen_manual.triggered.connect(self.show_help)

        # Non-assigned shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self).activated.connect(
            self.pressed_delete)  # Pressed Delete

        # Statusbar
        self.statusBarLabel = QtWidgets.QLabel()
        self.statusBar.addWidget(self.statusBarLabel)

        if len(sys.argv) >= 3:
            # startup arguments have been specified
            if sys.argv[1] == '-l':
                # Load specified graph
                self.load_graph(graphPath=sys.argv[2])
                # Delete the temporary graph
                os.remove(sys.argv[2])

    def gttr(self, variable, tfType=None):
        """
        :param variable: string/variable name, e.g. plotCanvas
        :param tfType: dict to current thin flow type
        :return: self.variable_tfType, e.g. self.plotCanvas_general or self.plotCanvas_spillback
        """
        if not tfType:
            tfType = self.currentTF
        return getattr(self, variable + '_' + tfType)

    def sttr(self, variable, tfType=None, value=None):
        """
        Sets variable_tfType var to value
        :param variable: string/variable name, e.g. plotCanvas
        :param value: value to set variable to
        :param tfType: dict to current thin flow type
        """
        if not tfType:
            tfType = self.currentTF
        setattr(self, variable + '_' + tfType, value)

    def tabSwitched(self, idx=None):
        """Slot for tab switching"""
        if idx is not None:
            self.currentTF = self.tfTypeList[idx]

    @staticmethod
    def init_graph():
        """
        Creates the initial network
        :return: network:initial network
        """
        network = nx.DiGraph()
        network.add_nodes_from(
            [('s', {'position': (-200, 0), 'label': 's'}), ('t', {'position': (200, 0), 'label': 't'})])
        network.graph['lastID'] = network.number_of_nodes() - 2  # Keep track of next nodes ID
        network.graph['inflowRate'] = 1  # Default inflowrate for new networks
        return network

    def init_app(self):  # former: init_graph_creation_app
        """Initialization of Tabs"""
        for tfType in self.tfTypeList:
            # Configure plotFrame to display plots of graphs
            self.sttr('plotFrameLayout', tfType, QtWidgets.QVBoxLayout())
            self.gttr('plotFrame', tfType).setLayout(self.gttr('plotFrameLayout', tfType))
            self.sttr('graphCreationCanvas', tfType, PlotCanvas(self.gttr('network', tfType), self,
                                                                stretchFactor=self.plotCanvasStretchFactor, onlyNTF=False,
                                                                type=tfType))  # Initialize PlotCanvas
            self.gttr('plotFrameLayout', tfType).addWidget(self.gttr('graphCreationCanvas', tfType))

            # Configure plotNTFFrame
            self.sttr('plotNTFFrameLayout', tfType, QtWidgets.QVBoxLayout())
            self.gttr('plotNTFFrame', tfType).setLayout(self.gttr('plotNTFFrameLayout', tfType))

            '''
            # Add empty graph to plotNTFCanvas to not destroy layout
            self.sttr('plotNTFCanvas', tfType, PlotNTFCanvas(nx.DiGraph(), self, intervalID=None,
                                                             stretchFactor=self.plotNTFCanvasStretchFactor,
                                                             showNoFlowEdges=self.showEdgesWithoutFlowCheckBox.isChecked(),
                                                             onlyNTF=True))
             '''
            self.sttr('plotNTFCanvas', tfType, None)
            self.sttr('NTFPlotList', tfType, None)

            # self.gttr('plotNTFFrameLayout', tfType).addWidget(self.gttr('plotNTFCanvas', tfType))

            # Configure plotAnimationFrame to display animation
            self.sttr('plotAnimationFrameLayout', tfType, QtWidgets.QVBoxLayout())
            self.gttr('plotAnimationFrame', tfType).setLayout(self.gttr('plotAnimationFrameLayout', tfType))
            self.sttr('plotAnimationCanvas', tfType, None)

            # Configure plotDiagramFrame to display edge in- and outflow, queue-size and node labels
            self.sttr('plotDiagramLayout', tfType, QtWidgets.QVBoxLayout())
            self.gttr('plotDiagramFrame', tfType).setLayout(self.gttr('plotDiagramLayout', tfType))
            self.sttr('plotDiagramCanvas', tfType, PlotValuesCanvas(callback=self.callback_plotvaluescanvas))
            self.gttr('plotDiagramLayout', tfType).addWidget(self.gttr('plotDiagramCanvas', tfType))

            # Configure plotQueueFrame to display queue of selected edge
            self.sttr('plotQueueFrameLayout', tfType, QtWidgets.QVBoxLayout())
            self.gttr('plotQueueFrame', tfType).setLayout(self.gttr('plotQueueFrameLayout', tfType))
            self.sttr('plotQueueCanvas', tfType, PlotQueueCanvas())
            self.gttr('plotQueueFrameLayout', tfType).addWidget(self.gttr('plotQueueCanvas', tfType))

        self.re_init_node_list()
        self.re_init_edge_list()

    # noinspection PyCallByClass,PyCallByClass
    def generate_animation_dialog(self):
        """Show dialog in which new animation range is selected"""
        lowerBoundInput, lowerGiven = QtWidgets.QInputDialog.getText(self, 'Enter the lower bound', 'Starting from:')
        if not lowerGiven:
            return
        upperBoundInput, upperGiven = QtWidgets.QInputDialog.getText(self, 'Enter the upper bound', 'Ending at:')
        if not upperGiven:
            return

        # Check if floats given
        try:
            float(lowerBoundInput)
            float(upperBoundInput)
        except ValueError:
            return

        self.generate_animation(lowerBoundInput, upperBoundInput)

    def generate_animation(self, lowerBoundInput, upperBoundInput):
        """Generates new animation"""
        if upperBoundInput == "":
            upperBoundInput = self.gttr('nashFlow').node_label('t', self.gttr('nashFlow').flowIntervals[-1][1]) \
                if self.gttr('nashFlow').flowIntervals[-1][1] < float('inf') \
                else Utilities.round_up(
                self.gttr('nashFlow').node_label('t', self.gttr('nashFlow').flowIntervals[-1][0]))

        lowerBound = float(lowerBoundInput) if lowerBoundInput != "" else 0
        upperBound = float(upperBoundInput)

        self.sttr('animationLowerBound', None, lowerBound)
        self.sttr('animationUpperBound', None, upperBound)

        self.update_diagrams()  # Update the value diagrams according to the new range

        self.gttr('timeSlider').setValue(0)  # Reset slider
        self.gttr('plotAnimationCanvas').reset_bounds(self.gttr('animationLowerBound'),
                                                      self.gttr('animationUpperBound'))
        self.gttr('plotQueueCanvas').update_information_callback(self.gttr('plotAnimationCanvas'))
        self.output("Generating animation")

    def update_node_display(self):
        """Update display of the properties of the currently focussed node self.graphCreationCanvas.focusNode, if existing"""
        if self.gttr('graphCreationCanvas').focusNode is not None:
            vertex = self.gttr('graphCreationCanvas').focusNode
            self.gttr('nodeNameLineEdit').setText(self.gttr('network').nodes[vertex]['label'])
            self.gttr('nodeXLineEdit').setText(str(round(self.gttr('network').nodes[vertex]['position'][0], 2)))
            self.gttr('nodeYLineEdit').setText(str(round(self.gttr('network').nodes[vertex]['position'][1], 2)))
        else:
            self.gttr('nodeNameLineEdit').setText("")
            self.gttr('nodeXLineEdit').setText("")
            self.gttr('nodeYLineEdit').setText("")

        self.setFocus()  # Focus has to leave LineEdits

    def update_node(self):
        """Update attributes of focusNode"""
        if self.gttr('graphCreationCanvas').focusNode is None:
            return

        nodeName = str(self.gttr('nodeNameLineEdit').text())
        XPos = str(self.gttr('nodeXLineEdit').text())
        YPos = str(self.gttr('nodeYLineEdit').text())
        if len(nodeName) > 0 and len(XPos) > 0 and len(YPos) > 0:
            vertex = self.gttr('graphCreationCanvas').focusNode
            if nodeName != self.gttr('network').nodes[vertex]['label']:
                self.gttr('network').nodes[vertex]['label'] = nodeName
                item = self.gttr('nodeToListItem')[vertex]
                self.gttr('nodeSelectionListWidget').takeItem(
                    self.gttr('nodeSelectionListWidget').row(item))  # Delete item
                self.add_node_to_list(vertex, self.currentTF)
                self.gttr('nodeSelectionListWidget').sortItems()  # Re-sort

            movedBool = (self.gttr('network').nodes[vertex]['position'] != (float(XPos), float(YPos)))
            self.gttr('network').nodes[vertex]['position'] = (float(XPos), float(YPos))

            self.gttr('graphCreationCanvas').update_nodes(moved=movedBool, color=True)  # Update UI
            if movedBool:
                self.gttr('graphCreationCanvas').update_edges(moved=movedBool)

    def delete_node(self):
        """Delete focusNode from network"""
        vertex = self.gttr('graphCreationCanvas').focusNode
        if vertex is None or vertex in ['s', 't']:
            return

        if vertex in self.gttr('network'):
            item = self.gttr('nodeToListItem')[vertex]
            index = self.gttr('nodeSelectionListWidget').row(item)
            self.gttr('nodeSelectionListWidget').takeItem(index)

            for edge in self.gttr('network').edges():
                if vertex in edge:
                    item = self.gttr('edgeToListItem')[edge]
                    index = self.gttr('edgeSelectionListWidget').row(item)
                    self.gttr('edgeSelectionListWidget').takeItem(index)

            self.gttr('graphCreationCanvas').update_nodes(removal=True, color=True)
            numberOfEdges = self.gttr('network').number_of_edges()
            self.gttr('network').remove_node(vertex)

            removedEdgeBool = (numberOfEdges > self.gttr('network').number_of_edges())
            self.gttr('graphCreationCanvas').focusNode = None

            if removedEdgeBool:
                self.gttr('graphCreationCanvas').update_edges(removal=True)

            self.update_node_display()  # Update UI

    def update_edge_display(self):
        """Update display of the properties of the currently focussed edge focusEdge, if existing"""
        edge = self.gttr('graphCreationCanvas').focusEdge
        if edge is not None:
            self.gttr('transitTimeLineEdit').setText(
                str(self.gttr('network')[edge[0]][edge[1]]['transitTime']))

            if self.currentTF == 'general':
                self.capacityLineEdit_general.setText(
                    str(self.gttr('network', 'general')[edge[0]][edge[1]]['outCapacity']))
            elif self.currentTF == 'spillback':
                self.inCapacityLineEdit_spillback.setText(
                    str(self.gttr('network', 'spillback')[edge[0]][edge[1]]['inCapacity']))
                self.outCapacityLineEdit_spillback.setText(
                    str(self.gttr('network', 'spillback')[edge[0]][edge[1]]['outCapacity']))
                self.storageLineEdit_spillback.setText(
                    str(self.gttr('network', 'spillback')[edge[0]][edge[1]]['storage']))
        else:
            self.gttr('transitTimeLineEdit').setText("")

            if self.currentTF == 'general':
                self.capacityLineEdit_general.setText("")
            elif self.currentTF == 'spillback':
                self.inCapacityLineEdit_spillback.setText("")
                self.outCapacityLineEdit_spillback.setText("")
                self.storageLineEdit_spillback.setText("")

        self.setFocus()  # Focus has to leave LineEdits

    def update_add_edge(self):
        """Add an edge or update attributes of focusEdge, if existing"""
        if self.gttr('graphCreationCanvas').focusEdge is None:
            return
        focusEdge = self.gttr('graphCreationCanvas').focusEdge

        tail = str(focusEdge[0])
        head = str(focusEdge[1])
        transitText = float(self.gttr('transitTimeLineEdit').text())

        if self.currentTF == 'general':
            capacityText = float(self.capacityLineEdit_general.text())
            if capacityText <= 0 or transitText < 0:
                return
        elif self.currentTF == 'spillback':
            inCapacityText = float(self.inCapacityLineEdit_spillback.text())
            outCapacityText = float(self.outCapacityLineEdit_spillback.text())
            storageText = float(self.storageLineEdit_spillback.text())
            if inCapacityText <= 0 or outCapacityText <= 0 or transitText < 0 or storageText <= 0:
                return
            elif tail == 's' and storageText != float('inf'):
                # This has to be satisfied
                return
            if storageText <= transitText * inCapacityText:
                # This is not allowed and is corrected automatically
                corrector = 0.1 if storageText / transitText > 0.1 else 0.1 * storageText / transitText
                inCapacityText = storageText / transitText - corrector
                self.inCapacityLineEdit_spillback.setText(str(inCapacityText))

        if self.gttr('network').has_edge(tail, head):
            # Update the edges attributes
            self.gttr('network')[tail][head]['transitTime'] = transitText
            if self.currentTF == 'general':
                self.gttr('network')[tail][head]['outCapacity'] = capacityText
            elif self.currentTF == 'spillback':
                self.gttr('network')[tail][head]['inCapacity'] = inCapacityText
                self.gttr('network')[tail][head]['outCapacity'] = outCapacityText
                self.gttr('network')[tail][head]['storage'] = storageText
            self.gttr('graphCreationCanvas').update_edges()
        else:
            # Add a new edge
            if self.currentTF == 'general':
                self.gttr('network').add_edge(tail, head, outCapacity=capacityText, transitTime=transitText,
                                              inCapacity=float('inf'), storage=float('inf'))
            elif self.currentTF == 'spillback':
                self.gttr('network').add_edge(tail, head, inCapacity=inCapacityText, outCapacity=outCapacityText,
                                              storage=storageText, transitTime=transitText)

            self.gttr('graphCreationCanvas').focusEdge = (tail, head)
            self.gttr('graphCreationCanvas').update_edges(added=True, color=True)
            self.gttr('graphCreationCanvas').update_nodes(color=True)
            self.add_edge_to_list((tail, head))
            self.gttr('edgeSelectionListWidget').sortItems()

        self.update_edge_display()  # Update UI

    def delete_edge(self):
        """Delete focusEdge from network"""
        edge = self.gttr('graphCreationCanvas').focusEdge
        if edge is None:
            return

        if self.gttr('network').has_edge(edge[0], edge[1]):
            item = self.gttr('edgeToListItem')[edge]
            index = self.gttr('edgeSelectionListWidget').row(item)
            self.gttr('edgeSelectionListWidget').takeItem(index)

            self.gttr('network').remove_edge(edge[0], edge[1])  # Deletion before update, as opposed to delete_node()
            self.gttr('graphCreationCanvas').update_edges(removal=True)

            self.gttr('graphCreationCanvas').focusEdge = None
            self.update_edge_display()  # Update UI

    def re_init_app(self, NoNewGraph=False):
        """
        Clears the graph creation frame for new graph creation
        :param NoNewGraph: (bool) - specify whether a new graph should be initiated or the old one kept
        """
        if not NoNewGraph:
            self.sttr('network', self.currentTF, self.init_graph())  # Reinstantiation of the CurrentGraph
            self.gttr('network', self.currentTF).graph['type'] = self.currentTF
            self.output("Clearing graph")

        # Reinitialization of graphCreationCanvas
        self.gttr('graphCreationCanvas').setParent(None)  # Drop graphCreationCanvas widget
        self.sttr('graphCreationCanvas', self.currentTF,
                  PlotCanvas(self.gttr('network'), self, self.plotCanvasStretchFactor, onlyNTF=False,
                             type=self.currentTF))
        self.gttr('plotFrameLayout').addWidget(
            self.gttr('graphCreationCanvas'))  # Add graphCreationCanvas-widget to application

        # Update UI
        self.update_node_display()
        self.update_edge_display()
        self.gttr('inflowLineEdit').setText(str(self.gttr('network').graph['inflowRate']))

        self.re_init_node_list()
        self.re_init_edge_list()

        #self.gttr('graphCreationCanvas').correct_edge_label_transangle()

    def re_init_nashflow_app(self):
        """Clears the nashflow tab for new nashflow computation"""
        self.gttr('computeIntervalPushButton').setEnabled(True)

        # Configure plotNTFFrame to display plots of NTF
        if self.gttr('plotNTFCanvas') is not None:
            self.gttr('plotNTFCanvas').setParent(None)
        self.sttr('plotNTFCanvas', None, None)
        self.sttr('NTFPlotList', None, [])

        self.gttr('intervalsListWidget').clear()

        # Configure plotAnimationFrame to display animation
        if self.gttr('plotAnimationCanvas') is not None:
            self.gttr('plotAnimationCanvas').setParent(None)

        self.sttr('animationUpperBound', None,
                  self.gttr('nashFlow').node_label('t', self.gttr('nashFlow').flowIntervals[-1][1]) \
                      if self.gttr('nashFlow').flowIntervals[-1][1] < float('inf') \
                      else Utilities.round_up(
                      self.gttr('nashFlow').node_label('t', self.gttr('nashFlow').flowIntervals[-1][0])))

        self.sttr('animationUpperBound', None, self.gttr('animationUpperBound') if self.gttr(
            'animationUpperBound') > 0 else 10)  # This can only happen when infinity is reached

        self.sttr('plotAnimationCanvas', None, PlotAnimationCanvas(nashflow=self.gttr('nashFlow'), interface=self,
                                                                   upperBound=self.gttr('animationUpperBound'),
                                                                   stretchFactor=self.plotCanvasStretchFactor))
        self.gttr('plotAnimationFrameLayout').addWidget(self.gttr('plotAnimationCanvas'))
        self.gttr('timeSlider').setMaximum(99)
        self.output("Generating animation")
        self.gttr('timeSlider').setValue(0)

        # Configure plotDiagramFrame
        if self.gttr('plotDiagramCanvas') is not None:
            self.gttr('plotDiagramCanvas').setParent(None)
        self.sttr('plotDiagramCanvas', None, PlotValuesCanvas(callback=self.callback_plotvaluescanvas))
        self.gttr('plotDiagramLayout').addWidget(self.gttr('plotDiagramCanvas'))

        # Configure plotQueueFrame
        if self.gttr('plotQueueCanvas') is not None:
            self.gttr('plotQueueCanvas').setParent(None)
        self.sttr('plotQueueCanvas', None, PlotQueueCanvas())
        self.gttr('plotQueueFrameLayout').addWidget(self.gttr('plotQueueCanvas'))
        self.gttr('plotQueueCanvas').update_information_callback(self.gttr('plotAnimationCanvas'))

        self.display_statistics()

    def display_statistics(self):
        """Display statistics"""
        nF = self.gttr('nashFlow')
        avgNodes, avgEdges = nF.get_stat_preprocessing()
        avgIPs, totalIPs = self.gttr('nashFlow').get_stat_solved_IPs()
        avgTime, totalTime = self.gttr('nashFlow').get_stat_time()

        nNodesStr = "#Nodes: " + str(nF.network.number_of_nodes())
        nEdgesStr = "#Edges: " + str(nF.network.number_of_edges())
        avgNodesStr = "Avg. #Deleted Nodes: " + str(avgNodes)
        avgEdgesStr = "Avg. #Deleted Edges: " + str(avgEdges)
        avgIPStr = "Avg. #IPs/Interval: " + str(avgIPs)
        totalIPStr = "Total #IPs: " + str(totalIPs)
        avgTimeStr = "Avg. Time/Interval: " + str(avgTime)
        totalTimeStr = "Total time: " + str(totalTime)

        stringList = [nNodesStr, nEdgesStr, avgIPStr, totalIPStr, avgTimeStr, totalTimeStr]
        statStr = " " + " | ".join(stringList)
        self.statusBarLabel.setText(statStr)

    def load_graph(self, graphPath=None):
        """
        Load graph instance from '.cg' file
        :param graphPath: If given, then function uses graphPath (must end in .cg) as path. Otherwise dialog gets opened
        :return: 
        """
        if not graphPath:
            dialog = QtWidgets.QFileDialog
            # noinspection PyCallByClass,PyCallByClass
            fopen = dialog.getOpenFileName(self, "Select File", self.defaultLoadSaveDir, "network files (*.cg)")
            fopen = fopen[0]
            if len(fopen) == 0:
                return
            fopen = str(fopen)
        else:
            fopen = graphPath

        # Read file         
        with open(fopen, 'rb') as f:
            network = pickle.load(f)
            self.sttr('network', network.graph['type'], network)
            self.tabWidget.setCurrentIndex(self.tfTypeList.index(network.graph['type']))
            self.gttr('tabWidget').setCurrentIndex(0)
        if not graphPath:
            self.defaultLoadSaveDir = os.path.dirname(fopen)
            self.save_config()

        self.output("Loading graph: " + str(fopen))
        self.re_init_app(NoNewGraph=True)

    # noinspection PyCallByClass
    def save_graph(self, graphPath=None):
        """
        Save graph instance to '.cg' file
        :param graphPath: If given, then save graph at path graphPath. Else a dialog is opened
        :return: 
        """
        self.gttr('network').graph['inflowRate'] = float(self.gttr('inflowLineEdit').text())

        if not graphPath:
            dialog = QtWidgets.QFileDialog
            fsave = dialog.getSaveFileName(self, "Select File", self.defaultLoadSaveDir, "network files (*.cg)")

            fsave = fsave[0]
            if len(fsave) == 0:
                return
            fsave = str(fsave)
        else:
            fsave = graphPath

        if not fsave.endswith('cg'):
            fsave += ".cg"

        if not graphPath:
            self.defaultLoadSaveDir = os.path.dirname(fsave)
            self.save_config()

        self.output("Saving graph: " + str(fsave))
        # Save network instance to file
        with open(fsave, 'wb') as f:
            pickle.dump(self.gttr('network'), f)

    # noinspection PyCallByClass
    def load_nashflow(self):
        """Load NashFlow instance from '.nf' file"""
        dialog = QtWidgets.QFileDialog
        fopen = dialog.getOpenFileName(self, "Select File", self.defaultLoadSaveDir, "Nashflow files (*.nf)")

        fopen = fopen[0]
        if len(fopen) == 0:
            return
        fopen = str(fopen)

        # Read file
        with open(fopen, 'rb') as f:
            self.nashFlow = pickle.load(f)
        self.defaultLoadSaveDir = os.path.dirname(fopen)
        self.save_config()
        self.output("Loading Nashflow: " + str(fopen))
        self.re_init_nashflow_app()
        self.add_intervals_to_list()

        self.tabWidget.setCurrentIndex(1)

    # noinspection PyCallByClass
    def save_nashflow(self):
        """Save Nashflow instance to '.nf' file"""
        dialog = QtWidgets.QFileDialog
        fsave = dialog.getSaveFileName(self, "Select File", self.defaultLoadSaveDir, "Nashflow files (*.nf)")

        fsave = fsave[0]
        if len(fsave) == 0:
            return
        fsave = str(fsave)

        if not fsave.endswith('nf'):
            fsave += ".nf"
        self.defaultLoadSaveDir = os.path.dirname(fsave)
        self.save_config()
        self.output("Saving Nashflow: " + str(fsave))

        # Save network instance to file
        with open(fsave, 'wb') as f:
            pickle.dump(self.gttr('nashFlow'), f)

    # noinspection PyCallByClass
    def select_output_directory(self):
        """Select output directory for nash flow computation"""
        defaultDir = self.outputDirectory
        dialog = QtWidgets.QFileDialog
        fselect = dialog.getExistingDirectory(self, "Select Directory", defaultDir)

        # fselect = fselect[0]
        if len(fselect) == 0:
            return
        fselect = str(fselect)
        self.output("Selecting output directory: " + str(fselect))
        for tfType in self.tfTypeList:
            self.gttr('outputDirectoryLineEdit', tfType).setText(fselect)
        self.outputDirectory = fselect

    # noinspection PyCallByClass
    def select_scip_binary(self):
        """Select scip binary"""
        defaultDir = '' if not os.path.isfile(self.scipFile) else os.path.dirname(self.scipFile)
        dialog = QtWidgets.QFileDialog
        fselect = dialog.getOpenFileName(self, "Select File", defaultDir)

        fselect = fselect[0]
        if len(fselect) == 0:
            return
        fselect = str(fselect)
        self.output("Selecting scip binary: " + str(fselect))
        for tfType in self.tfTypeList:
            self.gttr('scipPathLineEdit', tfType).setText(fselect)
        self.scipFile = fselect

    def load_config(self):
        """Try to load the config file"""
        self.configFile.add_section('Settings')
        self.configFile.set('Settings', 'outputdir', '')
        self.configFile.set('Settings', 'templatefile', '0')
        self.configFile.set('Settings', 'scippath', '')
        self.configFile.set('Settings', 'cleanup', 'True')
        self.configFile.set('Settings', 'intervals', '-1')
        self.configFile.set('Settings', 'defaultloadsavedir', '')
        self.configFile.set('Settings', 'timeoutactivated', 'True')
        self.configFile.set('Settings', 'timeoutTime', '300')

        try:
            self.configFile.read('config.cfg')
            self.outputDirectory = self.configFile.get('Settings', 'outputdir')
            self.templateFile = int(self.configFile.get('Settings', 'templatefile'))
            self.scipFile = self.configFile.get('Settings', 'scippath')
            self.cleanUpEnabled = (self.configFile.get('Settings', 'cleanup') == 'True')
            self.numberOfIntervals = self.configFile.get('Settings', 'intervals')
            self.defaultLoadSaveDir = self.configFile.get('Settings', 'defaultloadsavedir')
            self.timeoutActivated = (self.configFile.get('Settings', 'timeoutactivated') == 'True')
            timeoutTime = self.configFile.get('Settings', 'timeoutTime')

            for tfType in self.tfTypeList:
                self.gttr('outputDirectoryLineEdit', tfType).setText(self.outputDirectory)
                if tfType == 'general':
                    self.gttr('templateComboBox', tfType).setCurrentIndex(self.templateFile)
                elif tfType == 'spillback':
                    self.gttr('templateComboBox', tfType).setCurrentIndex(0)
                self.gttr('scipPathLineEdit', tfType).setText(self.scipFile)
                self.gttr('cleanUpCheckBox', tfType).setChecked(self.cleanUpEnabled)
                self.gttr('intervalsLineEdit', tfType).setText(self.numberOfIntervals)
                self.gttr('activateTimeoutCheckBox', tfType).setChecked(self.timeoutActivated)
                self.gttr('timeoutLineEdit', tfType).setText(timeoutTime)
            self.change_timeout_state()

            self.output("Loading config: Success")

        except Exception:
            self.output("Loading config: Failure")
            return

    def save_config(self):
        """Save the config file"""
        self.configFile.set('Settings', 'outputdir', self.outputDirectory)
        self.configFile.set('Settings', 'templatefile', str(self.templateFile))
        self.configFile.set('Settings', 'scippath', self.scipFile)
        self.configFile.set('Settings', 'cleanup', str(self.cleanUpEnabled))
        self.configFile.set('Settings', 'intervals', str(self.numberOfIntervals))
        self.configFile.set('Settings', 'defaultloadsavedir', self.defaultLoadSaveDir)
        self.configFile.set('Settings', 'timeoutactivated', str(self.timeoutActivated))
        timeoutTime = str(self.gttr('timeoutLineEdit').text())
        self.configFile.set('Settings', 'timeoutTime', timeoutTime)

        with open('config.cfg', 'w') as configfile:
            self.configFile.write(configfile)

        self.output("Saving config: config.cfg")

    # noinspection PyCallByClass
    def compute_nash_flow(self, nextIntervalOnly=False):
        """
        Computes a nash flow
        :param nextIntervalOnly: If this is True, only one interval(i.e. the next one) is computed
        """

        # Validate input
        returnCode = self.validate_input()
        if returnCode != 0:
            # Invalid input has been given
            # Spawn warning
            QtWidgets.QMessageBox.question(QtWidgets.QWidget(), 'Abort: Input error',
                                           self.get_error_message(returnCode),
                                           QtWidgets.QMessageBox.Ok)
            return

        # While computing it should not be possible to change showEdgesWithoutFlowCheckBox
        self.gttr('showEdgesWithoutFlowCheckBox').setEnabled(False)

        # Get remaining settings
        self.numberOfIntervals = self.gttr('intervalsLineEdit').text()
        self.templateFile = self.gttr(
            'templateComboBox').currentIndex()
        inflowRate = float(self.gttr('inflowLineEdit').text())

        self.save_config()  # Save config-settings to file
        self.gttr('tabWidget').setCurrentIndex(1)  # Switch to next tab
        timeout = -1 if not self.timeoutActivated else float(self.gttr('timeoutLineEdit').text())

        if not nextIntervalOnly:
            numberString = str(self.numberOfIntervals) if float(self.numberOfIntervals) != -1 else "all"
            self.output("Starting computation of " + numberString + " flow intervals")

            if self.currentTF == 'general':
                self.sttr('nashFlow', None,
                          NashFlow(self.gttr('network'), float(inflowRate), float(self.numberOfIntervals),
                                   self.outputDirectory,
                                   self.templateFile, self.scipFile, self.cleanUpEnabled, timeout))
            elif self.currentTF == 'spillback':
                self.sttr('nashFlow', None,
                          NashFlow_spillback(self.gttr('network'), float(inflowRate), float(self.numberOfIntervals),
                                             self.outputDirectory,
                                             self.templateFile, self.scipFile, self.cleanUpEnabled,
                                             timeout))
        else:
            self.output("Starting computation of next flow interval")
        self.gttr('nashFlow').run(nextIntervalOnly)

        self.output("Computation complete in " + "%.2f" % self.gttr('nashFlow').computationalTime + " seconds")

        self.re_init_nashflow_app()
        self.add_intervals_to_list()

        self.gttr('intervalsListWidget').setCurrentRow(0)
        self.slider_released()  # Update NTFs to display first NTF

        if self.gttr('nashFlow').infinityReached:
            self.gttr('computeIntervalPushButton').setEnabled(False)

        self.gttr('showEdgesWithoutFlowCheckBox').setEnabled(True)

    def compute_next_interval(self):
        """Computes next interval"""
        self.compute_nash_flow(nextIntervalOnly=(self.gttr('nashFlow') is not None))

    def add_intervals_to_list(self):
        """Adds NTF-intervals to the ListWidget"""
        for index, interval in enumerate(self.gttr('nashFlow').flowIntervals):
            intervalString = 'Interval ' + str(interval[2].id) + ': [' + str("%.2f" % interval[0]) + ',' + str(
                "%.2f" % interval[1]) + '['
            item = QtWidgets.QListWidgetItem(intervalString)
            item.setBackground(
                QtGui.QColor(self.gttr('plotAnimationCanvas').NTFColors[
                                 index % len(self.gttr('plotAnimationCanvas').NTFColors)]))
            self.gttr('intervalsListWidget').addItem(item)  # Add item to listWidget

            plot = PlotNTFCanvas(interval[2].shortestPathNetwork, self, intervalID=index,
                                 stretchFactor=self.plotCanvasStretchFactor,
                                 showNoFlowEdges=self.gttr('showEdgesWithoutFlowCheckBox').isChecked())
            # Note: this could create problems if user changes showEdgesWithoutFlowCheckBox between interval computations
            self.gttr('NTFPlotList').append(plot)  # Add NTF Plot to List

    def update_ntf_display(self):
        """Update plot of NTF corresponding to currently selected NTF in ListWidget"""
        rowID = self.gttr('intervalsListWidget').currentRow()
        lastViewPoint = None
        if rowID < 0:
            return

        normalFont, boldFont = QtGui.QFont(), QtGui.QFont()
        normalFont.setBold(False)
        boldFont.setBold(True)
        for row in range(self.gttr('intervalsListWidget').count()):
            item = self.gttr('intervalsListWidget').item(row)
            if row != rowID:
                item.setFont(normalFont)
            else:
                item.setFont(boldFont)

        if self.gttr('plotNTFCanvas') is not None:
            lastViewPoint = self.gttr('plotNTFCanvas').get_viewpoint()
            self.gttr('plotNTFCanvas').setParent(None)

        self.sttr('plotNTFCanvas', None, self.gttr('NTFPlotList')[rowID])
        self.gttr('plotNTFCanvas').set_viewpoint(viewPoint=lastViewPoint)  # Set viewpoint of plotNTF

        self.gttr('plotNTFFrameLayout').addWidget(self.gttr('plotNTFCanvas'))
        self.callback_plotvaluescanvas(self.gttr('nashFlow').flowIntervals[rowID][0], False)  # Update plotValues

    def slider_value_change(self):
        """Slot function for changes of slider val"""
        val = self.gttr('timeSlider').value()
        self.gttr('plotAnimationCanvas').time_changed(val)
        self.gttr('plotQueueCanvas').time_changed(val)
        currentTime = self.gttr('plotAnimationCanvas').get_time_from_tick(val)
        self.gttr('plotDiagramCanvas').change_vline_position(currentTime)
        self.gttr('currentSliderTimeLabel').setText("%.2f" % currentTime)

        self.update_current_queue_size_and_load(t=currentTime)

    def update_current_queue_size_and_load(self, t=None):
        if not t:
            val = self.gttr('timeSlider').value()
            t = self.gttr('plotAnimationCanvas').get_time_from_tick(val)
        edge = self.gttr('plotAnimationCanvas').focusEdge
        if edge is None:
            return
        v, w = edge
        queueSize, load = self.gttr('plotDiagramCanvas').get_approx_data(t)
        queueSizeLabel = "%.2f" % queueSize if queueSize != "N/A" else "N/A"
        if load == "N/A":
            loadLabel = "N/A"
        else:
            load = float(load)
            loadStorageRatio = load/self.gttr('network')[v][w]['storage'] * 100
            loadLabel = str(int(loadStorageRatio))

        self.gttr('currentQueueSizeLabel').setText(queueSizeLabel)

        if self.currentTF == 'spillback':
            self.currentLoadLabel_spillback.setText(loadLabel)

    def change_NTF_display(self, index):
        """
        Changes currently displayed NTF to index
        """
        lastViewPoint = None
        self.gttr('intervalsListWidget').setCurrentRow(index)
        if self.gttr('plotNTFCanvas') is not None:
            lastViewPoint = self.gttr('plotNTFCanvas').get_viewpoint()
            self.gttr('plotNTFCanvas').setParent(None)

        self.sttr('plotNTFCanvas', None, self.gttr('NTFPlotList')[index])
        self.gttr('plotNTFCanvas').set_viewpoint(viewPoint=lastViewPoint)
        self.gttr('plotNTFFrameLayout').addWidget(self.gttr('plotNTFCanvas'))

        rowID = self.gttr('intervalsListWidget').currentRow()

        normalFont, boldFont = QtGui.QFont(), QtGui.QFont()
        normalFont.setBold(False)
        boldFont.setBold(True)
        for row in range(self.gttr('intervalsListWidget').count()):
            item = self.gttr('intervalsListWidget').item(row)
            if row != rowID:
                item.setFont(normalFont)
            else:
                item.setFont(boldFont)

    def slider_released(self):
        """Slot function for slider release"""
        currentTime = self.gttr('plotAnimationCanvas').get_time_from_tick(self.gttr('timeSlider').value())
        # Adjust NTF if necessary
        for index, interval in enumerate(self.gttr('nashFlow').flowIntervals):
            lowerBound = interval[0]
            upperBound = interval[1]
            if lowerBound <= currentTime < upperBound:
                if self.gttr('intervalsListWidget').currentRow != index:
                    self.change_NTF_display(index)
                break

    def update_node_label_diagram(self):
        """Update diagram of node label of focusNode"""
        if self.gttr('plotAnimationCanvas').focusNode is None:
            return
        v = self.gttr('plotAnimationCanvas').focusNode

        lowerBound = self.gttr('animationLowerBound')
        upperBound = self.gttr('animationUpperBound')

        xValues = [0]
        yValues = [self.gttr('nashFlow').node_label(v, 0)]
        for interval in self.gttr('nashFlow').flowIntervals:
            if interval[1] < float('inf'):
                xValues.append(interval[1])  # append upperBound of each interval
                yValues.append(self.gttr('nashFlow').node_label(v, interval[1]))

        if upperBound > xValues[-1] and self.gttr('nashFlow').flowIntervals[-1][1] == float('inf'):
            xValues.append(upperBound)
            yValues.append(self.gttr('nashFlow').node_label(v, upperBound))
        self.gttr('plotDiagramCanvas').update_plot(lowerBound, upperBound, ["Earliest arrival time"], xValues, yValues)

    def update_edge_diagrams(self):
        """Update diagram of node label of focusNode"""
        if self.gttr('plotAnimationCanvas').focusEdge is None:
            return
        v, w = self.gttr('plotAnimationCanvas').focusEdge[0], self.gttr('plotAnimationCanvas').focusEdge[1]
        nF = self.gttr('nashFlow')
        network = nF.network
        lowerBound = self.gttr('animationLowerBound')
        upperBound = self.gttr('animationUpperBound')

        inflowXValues = list(network[v][w]['cumulativeInflow'].keys())
        inflowYValues = list(network[v][w]['cumulativeInflow'].values())

        if upperBound > inflowXValues[-1] and nF.infinityReached:
            lastInflow = network[v][w]['inflow'][next(reversed(network[v][w]['inflow']))]
            val = inflowYValues[-1] + (upperBound - inflowXValues[-1]) * lastInflow
            inflowXValues.append(upperBound)
            inflowYValues.append(val)

        outflowXValues = list(network[v][w]['cumulativeOutflow'].keys())
        outflowYValues = list(network[v][w]['cumulativeOutflow'].values())

        if upperBound > outflowXValues[-1] and nF.infinityReached:
            lastOutflow = network[v][w]['outflow'][next(reversed(network[v][w]['outflow']))]
            val = outflowYValues[-1] + (upperBound - outflowXValues[-1]) * lastOutflow
            outflowXValues.append(upperBound)
            outflowYValues.append(val)

        queueXValues = list(network[v][w]['queueSize'].keys())
        queueYValues = list(network[v][w]['queueSize'].values())

        if upperBound > queueXValues[-1] and nF.infinityReached:
            # Queue size stays constant or grows (but queue is never empty, if not already)
            lastQueueSize = queueYValues[-1]
            lastQueueSizeTime = queueXValues[-1]
            lastInflowInterval = next(reversed(network[v][w]['inflow']))
            lastInflow = network[v][w]['inflow'][lastInflowInterval]

            l, u = lastQueueSizeTime, upperBound
            lastSize = lastQueueSize
            for timeInterval, outflowVal in network[v][w]['outflow'].items():
                wTimeLower, wTimeUpper = timeInterval
                if wTimeUpper <= l:
                    # Not relevant
                    continue
                elif l <= wTimeUpper <= u:
                    lastSize = max(0, lastSize + (lastInflow - outflowVal) * (wTimeUpper - l))
                    l = wTimeUpper
                    queueXValues.append(l)
                    queueYValues.append(lastSize)
                elif l <= u < wTimeUpper:
                    lastSize = max(0, lastSize + (lastInflow - outflowVal) * (u - l))
                    l = u
                    queueXValues.append(u)
                    queueYValues.append(lastSize)
                    break

        # Compute load=cumInflow - cumOutflow diagram
        # Interpolate values
        mRange = min(inflowXValues[-1], outflowXValues[-1])
        xUnion = list(set(inflowXValues).union(set(outflowXValues)))
        xUnion.sort()
        loadXValues = [x for x in xUnion if x <= mRange]
        loadYValues = []
        for x in loadXValues:
            # Compute inflow y
            for j in range(len(inflowXValues) - 1):
                if inflowXValues[j] <= x <= inflowXValues[j + 1]:
                    if x == inflowXValues[j + 1]:
                        y_i = inflowYValues[j + 1]
                        break
                    else:
                        m = float(inflowYValues[j + 1] - inflowYValues[j]) / (inflowXValues[j + 1] - inflowXValues[j])
                        y_i = inflowYValues[j] + (x - inflowXValues[j]) * m
                        break
            # Compute outflow y
            for j in range(len(outflowXValues) - 1):
                if outflowXValues[j] <= x <= outflowXValues[j + 1]:
                    if x == outflowXValues[j + 1]:
                        y_o = outflowYValues[j + 1]
                        break
                    else:
                        m = float(outflowYValues[j + 1] - outflowYValues[j]) / (
                                outflowXValues[j + 1] - outflowXValues[j])
                        y_o = outflowYValues[j] + (x - outflowXValues[j]) * m
                        break
            y = y_i - y_o
            loadYValues.append(y)

        # Display storage horizontal line
        if self.currentTF == 'general' or self.gttr('network')[v][w]['storage'] == float('inf'):
            storage = None
        else:
            storage = self.gttr('network')[v][w]['storage']

        self.gttr('plotDiagramCanvas').update_plot(lowerBound, upperBound,
                                                   ["Cumulative Inflow", "Cumulative Outflow", "Queue size", "Load"],
                                                   inflowXValues,
                                                   inflowYValues, storage, (outflowXValues, outflowYValues),
                                                   (queueXValues, queueYValues), (loadXValues, loadYValues))

    def update_diagrams(self):
        """Update diagrams of focusEdge or focusNode, depending on whats selected"""
        if self.gttr('plotAnimationCanvas').focusEdge is not None:
            self.update_edge_diagrams()
        elif self.gttr('plotAnimationCanvas').focusNode is not None:
            self.update_node_label_diagram()

    def change_cleanup_state(self):
        """Active/Deactive cleanup"""
        self.cleanUpEnabled = self.gttr('cleanUpCheckBox').isChecked()
        for tfType in self.tfTypeList:
            self.gttr('cleanUpCheckBox', tfType).setChecked(self.cleanUpEnabled)

    def pressed_delete(self):
        """Slot for DEL Key"""
        if self.gttr('graphCreationCanvas').focusNode is not None:
            self.delete_node()
        elif self.gttr('graphCreationCanvas').focusEdge is not None:
            self.delete_edge()

    def set_new_time_manually(self):
        """Set animation timepoint manually"""
        if self.gttr('setTimeLineEdit').text() == "":
            return
        else:
            val = float(self.gttr('setTimeLineEdit').text())
            self.callback_plotvaluescanvas(xVal=val, updateNTF=True)

    def callback_plotvaluescanvas(self, xVal, updateNTF=True):
        """
        Callback function for plotValuesCanvas
        :param xVal: selected time
        :param updateNTF: if true, then the NTF plot has to be updated
        """
        xVal = float("%.2f" % xVal)

        valueTol = 1e-2
        if not (self.gttr('plotAnimationCanvas').timePoints[0] <= xVal <= self.gttr('plotAnimationCanvas').timePoints[
            -1]):
            return

        try:
            # Check if there already exists a timepoint which is sufficiently close
            xVal = next(
                t for t in self.gttr('plotAnimationCanvas').timePoints if Utilities.is_eq_tol(t, xVal, valueTol))
        except StopIteration:
            # Add the time point
            self.gttr('plotAnimationCanvas').add_time(xVal)
            self.gttr('timeSlider').setMaximum(self.gttr('timeSlider').maximum() + 1)

        self.gttr('timeSlider').setValue(self.gttr('plotAnimationCanvas').timePoints.index(xVal))
        if updateNTF:
            self.slider_released()

    # noinspection PyCallByClass
    def export_diagram(self):
        """Export diagram to PDF or PGF"""
        fileType = 'pdf' if self.gttr('exportComboBox').currentIndex() == 0 else 'pgf'
        dialog = QtWidgets.QFileDialog
        fsave = dialog.getSaveFileName(self, "Select File", self.defaultLoadSaveDir,
                                       fileType + " files (*." + fileType + ")")

        fsave = fsave[0]
        if len(fsave) == 0:
            return
        fsave = str(fsave)

        if not fsave.endswith('.' + fileType):
            fsave += '.' + fileType

        self.defaultLoadSaveDir = os.path.dirname(fsave)
        self.save_config()
        self.output("Exporting diagram: " + str(fsave))

        self.gttr('plotDiagramCanvas').export(path=fsave)

    # noinspection PyCallByClass
    def export_animation(self):
        """Export animation to mp4. This requires FFMPEG."""
        fileType = 'mp4'
        dialog = QtWidgets.QFileDialog
        fsave = dialog.getSaveFileName(self, "Select File", self.defaultLoadSaveDir,
                                       fileType + " files (*." + fileType + ")")

        fsave = fsave[0]
        if len(fsave) == 0:
            return
        fsave = str(fsave)

        if not fsave.endswith('.' + fileType):
            fsave += '.' + fileType
        self.output("Exporting animation video: " + str(fsave))
        self.gttr('plotAnimationCanvas').export(path=fsave)

    def update_plotanimationcanvas_focusselection(self):
        """Update labels in Tab1 when focus changes"""
        if self.gttr('plotAnimationCanvas').focusNode is not None:
            self.gttr('currentFocusLabel').setText(str(self.gttr('plotAnimationCanvas').focusNode))
            self.gttr('currentCapacityLabel').setText("N/A")
            self.gttr('currentTransitTimeLabel').setText("N/A")
            self.gttr('plotQueueCanvas').change_focusEdge(None, None)
        elif self.gttr('plotAnimationCanvas').focusEdge is not None:
            v, w = self.gttr('plotAnimationCanvas').focusEdge
            self.gttr('currentFocusLabel').setText(
                str((self.gttr('nashFlow').network.nodes[v]['label'], self.gttr('nashFlow').network.nodes[w]['label'])))
            self.gttr('currentCapacityLabel').setText(str(self.gttr('nashFlow').network[v][w]['outCapacity']))
            self.gttr('currentTransitTimeLabel').setText(str(self.gttr('nashFlow').network[v][w]['transitTime']))
            self.gttr('plotQueueCanvas').change_focusEdge(v, w)
            self.update_current_queue_size_and_load()
        else:
            self.gttr('currentFocusLabel').setText("N/A")
            self.gttr('currentCapacityLabel').setText("N/A")
            self.gttr('currentTransitTimeLabel').setText("N/A")
            self.gttr('plotQueueCanvas').change_focusEdge(None, None)

    def open_tfc(self, moveGraph=None):
        """
        Opens ThinFlowComputation Tool
        :param moveGraph: network that should be moved, None if not specified
        """
        if not moveGraph:
            # Just open the application
            cmd = ['python3', 'thinFlow_mainControl.py']
        else:
            # Open the application with a certain graph
            # Save the graph
            tmpDir = gettempdir()
            tmpFileName = Utilities.get_time()
            tmpFilePath = os.path.join(tmpDir, tmpFileName)
            self.save_graph(graphPath=tmpFilePath)
            tmpFilePathArgument = tmpFilePath + '.cg'

            cmd = ['python3', 'thinFlow_mainControl.py', '-l', tmpFilePathArgument]

        def open_tfc_thread():
            self.proc = subprocess.Popen(cmd)
            self.proc.communicate()

        thread = threading.Thread(target=open_tfc_thread)
        thread.start()

    def move_to_tfc(self):
        self.open_tfc(moveGraph=self.gttr('network'))

    def change_no_flow_show_state(self):
        """Show/Hide edges without flow in each NTF Plot"""
        for NTF in self.gttr('NTFPlotList'):
            NTF.change_edge_show_status(show=self.gttr('showEdgesWithoutFlowCheckBox').isChecked())

    def update_focus_node(self):
        """Select new focusNode"""
        self.gttr('graphCreationCanvas').focusEdge = None
        index = self.gttr('nodeSelectionListWidget').currentRow()
        item = self.gttr('nodeSelectionListWidget').item(index)
        node = list(self.gttr('nodeToListItem').keys())[list(self.gttr('nodeToListItem').values()).index(item)]
        self.gttr('graphCreationCanvas').focusNode = node
        self.gttr('graphCreationCanvas').update_nodes(color=True)
        self.gttr('graphCreationCanvas').update_edges(color=True)
        self.update_node_display()
        self.update_edge_display()

    def update_focus_edge(self):
        """Select new focusEdge"""
        self.gttr('graphCreationCanvas').focusNode = None
        index = self.gttr('edgeSelectionListWidget').currentRow()
        item = self.gttr('edgeSelectionListWidget').item(index)
        edge = list(self.gttr('edgeToListItem').keys())[list(self.gttr('edgeToListItem').values()).index(item)]
        self.gttr('graphCreationCanvas').focusEdge = edge
        self.gttr('graphCreationCanvas').update_nodes(color=True)
        self.gttr('graphCreationCanvas').update_edges(color=True)
        self.update_node_display()
        self.update_edge_display()

    def re_init_node_list(self):
        """Clear and fill node list"""
        for tfType in self.tfTypeList:
            self.gttr('nodeSelectionListWidget', tfType).clear()
            self.sttr('nodeToListItem', tfType, dict())
            for node in self.gttr('network', tfType).nodes():
                self.add_node_to_list(node, tfType)
            self.gttr('nodeSelectionListWidget', tfType).sortItems()

    def re_init_edge_list(self):
        """Clear and fill edge list"""
        for tfType in self.tfTypeList:
            self.gttr('edgeSelectionListWidget', tfType).clear()
            self.sttr('edgeToListItem', tfType, dict())
            for edge in self.gttr('network', tfType).edges():
                self.add_edge_to_list(edge, tfType)
            self.gttr('edgeSelectionListWidget', tfType).sortItems()

    def add_node_to_list(self, node, tfType=None):
        """
        Add single node to list
        :param node: node which will be added to ListWidget of tfType
        :param tfType: dict containing information of current thinflowType
        """
        if not tfType:
            tfType = self.currentTF

        nodeString = 'Node ' + str(node) + ': ' + self.gttr('network', tfType).nodes[node]['label']
        item = QtWidgets.QListWidgetItem(nodeString)
        self.gttr('nodeToListItem', tfType)[node] = item
        self.gttr('nodeSelectionListWidget', tfType).addItem(item)  # Add item to listWidget

    def add_edge_to_list(self, edge, tfType=None):
        """
        Add single edge to list
        :param edge: edge which will be added to ListWidget of tfType
        :param tfType: dict containing information of current thinflowType
        """
        if not tfType:
            tfType = self.currentTF

        v, w = edge
        edgeString = 'Edge: ' + str(
            (self.gttr('network', tfType).nodes[v]['label'], self.gttr('network', tfType).nodes[w]['label']))
        item = QtWidgets.QListWidgetItem(edgeString)
        self.gttr('edgeToListItem')[edge] = item
        self.gttr('edgeSelectionListWidget', tfType).addItem(item)

    def change_timeout_state(self):
        """Activate/Deactivate Timeout"""
        self.timeoutActivated = self.gttr('activateTimeoutCheckBox').isChecked()
        for tfType in self.tfTypeList:
            self.gttr('activateTimeoutCheckBox', tfType).setChecked(self.timeoutActivated)
            self.gttr('timeoutLabel', tfType).setEnabled(self.timeoutActivated)
            self.gttr('timeoutLineEdit', tfType).setEnabled(self.timeoutActivated)

    def play_animation(self):
        """Slot to play animation"""

        def animate(FPS=4):
            """
            Animation function called by thread
            :param FPS: frames per second
            """
            currentIntervalIndex = self.gttr('plotAnimationCanvas').get_flowinterval_index_from_tick(
                self.gttr('timeSlider').value())
            self.change_NTF_display(currentIntervalIndex)

            get_next_bound = lambda currentIntervalIndex: self.gttr('nashFlow').flowIntervals[currentIntervalIndex + 1][
                0] \
                if currentIntervalIndex + 1 < len(self.gttr('nashFlow').flowIntervals) else float('inf')

            nextBound = get_next_bound(currentIntervalIndex)
            lastTime = self.gttr('plotAnimationCanvas').get_time_from_tick(self.gttr('timeSlider').value())
            while self.gttr('animationRunning') and self.gttr('timeSlider').value() < self.gttr('timeSlider').maximum():
                time.sleep(1 / float(FPS))
                self.gttr('timeSlider').setValue(self.gttr('timeSlider').value() + 1)
                if self.gttr('plotAnimationCanvas').get_time_from_tick(self.gttr('timeSlider').value()) >= nextBound:
                    currentIntervalIndex += 1
                    self.change_NTF_display(currentIntervalIndex)
                    nextBound = get_next_bound(currentIntervalIndex)

                if self.gttr('timeSlider').value() >= 1 and lastTime != self.gttr(
                        'plotAnimationCanvas').get_time_from_tick(
                    self.gttr('timeSlider').value() - 1):
                    # Necessary to check, as user could click somewhere on the slider
                    currentIntervalIndex = self.gttr('plotAnimationCanvas').get_flowinterval_index_from_tick(
                        self.gttr('timeSlider').value())
                    self.change_NTF_display(currentIntervalIndex)
                    nextBound = get_next_bound(currentIntervalIndex)
                lastTime = self.gttr('plotAnimationCanvas').get_time_from_tick(self.gttr('timeSlider').value())

            self.sttr('animationRunning', None, False)

        if not self.gttr('animationRunning'):
            self.output("Starting animation")
            self.sttr('animationRunning', None, True)
            t = threading.Thread(target=animate)
            t.daemon = True  # Enforcing that thread gets killed if main exits
            t.start()

    def pause_animation(self):
        """Slot to pause animation"""
        self.output("Pausing animation")
        self.sttr('animationRunning', None, False)

    def stop_animation(self):
        """Slot to stop animation and jump to beginning"""
        self.sttr('animationRunning', None, False)
        self.output("Stopping animation")
        time.sleep(0.5)

        self.gttr('timeSlider').setValue(0)

    def output(self, txt):
        """
        Write to log
        :param txt: will be written to log
        """
        currentTime = Utilities.get_time_for_log()
        logText = currentTime + " - " + txt
        for tfType in self.tfTypeList:
            self.gttr('logPlainTextEdit', tfType=tfType).appendPlainText(logText)

    @staticmethod
    def show_help():
        """Open thesis to display manual"""
        os.system('xdg-open documentation/thesis.pdf')

    def validate_input(self):
        """Checks whether network satisfies certain conditions, return respective error code if necessary"""
        network = self.gttr('network')
        if network.in_edges('s'):
            # Network may not contain edges going into s
            return 1
        elif network.out_edges('t'):
            # Edges going out from t
            return 2
        for (v, d) in network.in_degree():
            if d == 0 and v != 's':
                # Non-reachable node found
                return 3
        if min(nx.get_edge_attributes(network, 'outCapacity').values()) <= 0:
            # Wrong capacity attribute
            return 4

        if self.currentTF == 'spillback':
            if min(nx.get_edge_attributes(network, 'inCapacity').values()) <= 0:
                return 4

        try:
            # Try to find a cycle in the network
            nx.find_cycle(network)
            return 5
        except nx.exception.NetworkXNoCycle:
            pass

        if self.currentTF == 'spillback':
            try:
                for e in network.out_edges('s'):
                    (v, w) = e
                    if network[v][w]['inCapacity'] < float(
                            self.inflowLineEdit_spillback.text()):
                        return 6
            except KeyError:
                pass

        return 0

    @staticmethod
    def get_error_message(errorCode):
        errorDescription = {
            1: "Source 's' should not have incoming edges.",
            2: "Sink 't' should not have outgoing edges.",
            3: "All nodes have to be reachable from 's'.",
            4: "Edge capacities.",
            5: "Network contains a cycle.",
            6: "Incapacities of all outgoing edges from sink 's' have to be greater than network inflow-rate."
        }

        return errorDescription[errorCode]


