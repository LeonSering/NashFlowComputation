# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         thinFlow_application.py
# Description:  Interface class; controlling signals/slots & communication between widgets
# ===========================================================================


from PyQt5 import QtGui, QtCore, QtWidgets
import configparser
import os
import pickle
import sys
from shutil import rmtree
from copy import deepcopy
import networkx as nx
from warnings import filterwarnings
from tempfile import gettempdir
import subprocess
import threading

from source.plotCanvasClass import PlotCanvas
from source.plotNTFCanvasClass import PlotNTFCanvas
from source.ui import thinFlow_mainWdw
from source.utilitiesClass import Utilities
from source.application import Interface as app_Interface
from source.flowIntervalClass import FlowInterval
from source.flowIntervalClass_spillback import FlowInterval_spillback

# =======================================================================================================================
filterwarnings('ignore')  # For the moment: ignore warnings as pyplot.hold is deprecated
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)  # This is necessary if threads access the GUI


class Interface(QtWidgets.QMainWindow, thinFlow_mainWdw.Ui_MainWindow):
    """Controls GUI"""

    def __init__(self):
        """Initialization of Class and GUI"""
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        # Scaling factors of frames, to avoid distortion
        self.plotCanvasStretchFactor = float(self.plotFrame_general.width()) / self.plotFrame_general.height()
        self.plotNTFCanvasStretchFactor = float(self.plotNTFFrame_general.width()) / self.plotNTFFrame_general.height()

        self.tfTypeList = ['general', 'spillback']
        self.currentTF = self.tfTypeList[self.tabWidget.currentIndex()]  # Currently selected tab information

        # Init graphs
        for tfType in self.tfTypeList:
            self.sttr('network', tfType, app_Interface.init_graph())

        self.init_app()
        self.inflowLineEdit.setText('1')  # Default value

        # Config defaults
        self.outputDirectory = ''
        self.templateFile = 0  # 0,1,2 for three algorithms from thesis
        self.scipFile = ''
        self.timeoutActivated = False
        self.defaultLoadSaveDir = ''
        self.cleanUpEnabled = True

        self.configFile = configparser.RawConfigParser()  # This is the parser, not to confuse with the actual config.txt File, which cannot be specified

        # Initializations
        self.load_config()  # Try to load configuration file

        # Signal configuration
        self.tabWidget.currentChanged.connect(self.tabSwitched)

        for tfType in self.tfTypeList:
            self.gttr('updateNodeButton', tfType).clicked.connect(self.update_node)
            self.gttr('deleteNodeButton', tfType).clicked.connect(self.delete_node)
            self.gttr('updateEdgeButton', tfType).clicked.connect(self.update_add_edge)
            self.gttr('deleteEdgeButton', tfType).clicked.connect(self.delete_edge)
            self.gttr('nodeSelectionListWidget', tfType).clicked.connect(self.update_focus_node)
            self.gttr('edgeSelectionListWidget', tfType).clicked.connect(self.update_focus_edge)
            self.gttr('computeThinFlowPushButton', tfType).clicked.connect(self.compute_NTF)
            self.gttr('resettingSwitchButton', tfType).clicked.connect(self.change_resetting)

            # Keyboard shortcuts
            self.gttr('capacityLineEdit', tfType).returnPressed.connect(self.update_add_edge)
            self.gttr('nodeNameLineEdit', tfType).returnPressed.connect(self.update_node)
            self.gttr('nodeXLineEdit', tfType).returnPressed.connect(self.update_node)
            self.gttr('nodeYLineEdit', tfType).returnPressed.connect(self.update_node)

        # Spillback-only signals
        self.boundLineEdit_spillback.returnPressed.connect(self.update_add_edge)

        # General signals
        self.outputDirectoryPushButton.clicked.connect(self.select_output_directory)
        self.scipPathPushButton.clicked.connect(self.select_scip_binary)
        self.cleanUpCheckBox.clicked.connect(self.change_cleanup_state)
        self.showEdgesWithoutFlowCheckBox.clicked.connect(self.change_no_flow_show_state)
        self.activateTimeoutCheckBox.clicked.connect(self.change_timeout_state)
        self.actionNew_graph.triggered.connect(self.re_init_app)
        self.actionLoad_graph.triggered.connect(self.load_graph)
        self.actionSave_graph.triggered.connect(self.save_graph)
        self.actionExit.triggered.connect(QtWidgets.QApplication.quit)
        self.actionLoad_Thinflow.triggered.connect(self.load_thinflow)
        self.actionSave_Thinflow.triggered.connect(self.save_thinflow)
        self.actionOpen_NashFlowComputation.triggered.connect(self.open_nfc)
        self.actionMove_current_graph_to_NashFlowComputation.triggered.connect(self.move_to_nfc)
        # TO BE DONE LATER
        # self.actionOpen_manual.triggered.connect(self.show_help)

        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self).activated.connect(
            self.pressed_delete)  # Pressed Delete

        self.tabWidget.setCurrentIndex(0)  # Show General Tab

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

    def init_app(self):  # former: init_graph_creation_app
        """Initialization of Tabs"""
        for tfType in self.tfTypeList:
            # Configure plotFrame to display plots of graphs
            self.sttr('plotFrameLayout', tfType, QtWidgets.QVBoxLayout())
            self.gttr('plotFrame', tfType).setLayout(self.gttr('plotFrameLayout', tfType))
            self.sttr('graphCreationCanvas', tfType, PlotCanvas(self.gttr('network', tfType), self,
                                                                stretchFactor=self.plotCanvasStretchFactor, onlyNTF=True,
                                                                type=tfType))  # Initialize PlotCanvas
            self.gttr('plotFrameLayout', tfType).addWidget(self.gttr('graphCreationCanvas', tfType))

            # Configure plotNTFFrame
            self.sttr('plotNTFFrameLayout', tfType, QtWidgets.QVBoxLayout())
            self.gttr('plotNTFFrame', tfType).setLayout(self.gttr('plotNTFFrameLayout', tfType))

            # Add empty graph to plotNTFCanvas to not destroy layout
            self.sttr('plotNTFCanvas', tfType, PlotNTFCanvas(nx.DiGraph(), self, intervalID=None,
                                                             stretchFactor=self.plotNTFCanvasStretchFactor,
                                                             showNoFlowEdges=self.showEdgesWithoutFlowCheckBox.isChecked(),
                                                             onlyNTF=True))
            self.gttr('plotNTFFrameLayout', tfType).addWidget(self.gttr('plotNTFCanvas', tfType))

        self.re_init_node_list()
        self.re_init_edge_list()

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

        nodeString = 'Node ' + str(node) + ': ' + self.gttr('network', tfType).node[node]['label']
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
            (self.gttr('network', tfType).node[v]['label'], self.gttr('network', tfType).node[w]['label']))
        item = QtWidgets.QListWidgetItem(edgeString)
        self.gttr('edgeToListItem')[edge] = item
        self.gttr('edgeSelectionListWidget', tfType).addItem(item)

    def load_config(self):
        """Try to load the config file"""
        self.configFile.add_section('Settings')
        self.configFile.set('Settings', 'outputdir', '')
        self.configFile.set('Settings', 'templatefile', '0')
        self.configFile.set('Settings', 'scippath', '')
        self.configFile.set('Settings', 'cleanup', 'True')
        self.configFile.set('Settings', 'defaultloadsavedir', '')
        self.configFile.set('Settings', 'timeoutactivated', 'True')

        try:
            self.configFile.read('thinFlow_config.cfg')

            self.outputDirectory = self.configFile.get('Settings', 'outputdir')
            self.outputDirectoryLineEdit.setText(self.outputDirectory)
            self.templateFile = int(self.configFile.get('Settings', 'templatefile'))
            self.templateComboBox.setCurrentIndex(self.templateFile)
            self.scipFile = self.configFile.get('Settings', 'scippath')
            self.scipPathLineEdit.setText(self.scipFile)

            self.cleanUpEnabled = (self.configFile.get('Settings', 'cleanup') == 'True')
            self.cleanUpCheckBox.setChecked(self.cleanUpEnabled)

            self.defaultLoadSaveDir = self.configFile.get('Settings', 'defaultloadsavedir')

            self.timeoutActivated = (self.configFile.get('Settings', 'timeoutactivated') == 'True')
            self.activateTimeoutCheckBox.setChecked(self.timeoutActivated)
            self.change_timeout_state()

        except Exception:
            return

    def change_timeout_state(self):
        """Activate/Deactivate Timeout"""
        self.timeoutActivated = self.activateTimeoutCheckBox.isChecked()
        self.timeoutLabel.setEnabled(self.timeoutActivated)
        self.timeoutLineEdit.setEnabled(self.timeoutActivated)

    def update_node(self):
        """Update attributes of focusNode"""
        if self.gttr('graphCreationCanvas').focusNode is None:
            return

        nodeName = str(self.gttr('nodeNameLineEdit').text())
        XPos = str(self.gttr('nodeXLineEdit').text())
        YPos = str(self.gttr('nodeYLineEdit').text())
        if len(nodeName) > 0 and len(XPos) > 0 and len(YPos) > 0:
            vertex = self.gttr('graphCreationCanvas').focusNode
            if nodeName != self.gttr('network').node[vertex]['label']:
                self.gttr('network').node[vertex]['label'] = nodeName
                item = self.gttr('nodeToListItem')[vertex]
                self.self.gttr('nodeSelectionListWidget').takeItem(
                    self.gttr('nodeSelectionListWidget').row(item))  # Delete item
                self.add_node_to_list(vertex, self.currentTF)
                self.self.gttr('nodeSelectionListWidget').sortItems()  # Re-sort

            movedBool = (self.gttr('network').node[vertex]['position'] != (float(XPos), float(YPos)))
            self.gttr('network').node[vertex]['position'] = (float(XPos), float(YPos))

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

    def update_node_display(self):
        """Update display of the properties of the currently focussed node self.graphCreationCanvas.focusNode, if existing"""
        if self.gttr('graphCreationCanvas').focusNode is not None:
            vertex = self.gttr('graphCreationCanvas').focusNode
            self.gttr('nodeNameLineEdit').setText(self.gttr('network').node[vertex]['label'])
            self.gttr('nodeXLineEdit').setText(str(round(self.gttr('network').node[vertex]['position'][0], 2)))
            self.gttr('nodeYLineEdit').setText(str(round(self.gttr('network').node[vertex]['position'][1], 2)))
        else:
            self.gttr('nodeNameLineEdit').setText("")
            self.gttr('nodeXLineEdit').setText("")
            self.gttr('nodeYLineEdit').setText("")

        self.setFocus()  # Focus has to leave LineEdits

    def update_edge_display(self):
        """Update display of the properties of the currently focussed edge focusEdge, if existing"""
        edge = self.gttr('graphCreationCanvas').focusEdge
        if edge is not None:
            self.gttr('capacityLineEdit').setText(
                str(self.gttr('network')[edge[0]][edge[1]]['outCapacity']))

            if self.currentTF == 'spillback':
                self.boundLineEdit_spillback.setText(str(self.gttr('network')[edge[0]][edge[1]]['TFC']['inflowBound']))

        else:
            self.gttr('capacityLineEdit').setText("")

            if self.currentTF == 'spillback':
                self.boundLineEdit_spillback.setText("")

        self.setFocus()  # Focus has to leave LineEdits

        self.adjust_resettingSwitchButton(edge)

    def update_add_edge(self):
        """Update attributes of focusEdge, if existing"""
        if self.gttr('graphCreationCanvas').focusEdge is None:
            return
        focusEdge = self.gttr('graphCreationCanvas').focusEdge
        capacityText = float(self.gttr('capacityLineEdit').text())
        boundText = float(self.boundLineEdit_spillback.text()) if self.currentTF == 'spillback' else None

        tail = str(focusEdge[0])
        head = str(focusEdge[1])

        if capacityText <= 0:
            # This is not allowed
            return
        if boundText is not None and boundText <= 0:
            # Not allowed
            return

        if self.gttr('network').has_edge(tail, head):
            # Update the edges attributes
            self.gttr('network')[tail][head]['outCapacity'] = capacityText
            if boundText is not None:
                self.gttr('network', 'spillback')[tail][head]['TFC']['inflowBound'] = boundText
            self.gttr('graphCreationCanvas').update_edges()
        else:
            # Add a new edge
            if boundText is not None:
                self.gttr('network').add_edge(tail, head, capacity=capacityText, resettingEnabled=False)
            else:
                self.gttr('network').add_edge(tail, head, capacity=capacityText, inflowBound=boundText,
                                              resettingEnabled=False)
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

    def select_output_directory(self):
        """Select output directory for thin flow computation"""
        defaultDir = self.outputDirectory
        dialog = QtWidgets.QFileDialog
        fselect = dialog.getExistingDirectory(self, "Select Directory", defaultDir)

        #fselect = fselect[0]
        if len(fselect) == 0:
            return
        fselect = str(fselect)
        self.outputDirectoryLineEdit.setText(fselect)
        self.outputDirectory = fselect

    def select_scip_binary(self):
        """Select scip binary"""
        defaultDir = '' if not os.path.isfile(self.scipFile) else os.path.dirname(self.scipFile)
        dialog = QtWidgets.QFileDialog
        fselect = dialog.getOpenFileName(self, "Select File", defaultDir)

        fselect = fselect[0]
        if len(fselect) == 0:
            return
        fselect = str(fselect)
        self.scipPathLineEdit.setText(fselect)
        self.scipFile = fselect

    def save_config(self):
        """Save the config file"""
        self.configFile.set('Settings', 'outputdir', self.outputDirectory)
        self.configFile.set('Settings', 'templatefile', self.templateFile)
        self.configFile.set('Settings', 'scippath', self.scipFile)
        self.configFile.set('Settings', 'cleanup', self.cleanUpEnabled)
        self.configFile.set('Settings', 'defaultloadsavedir', self.defaultLoadSaveDir)
        self.configFile.set('Settings', 'timeoutactivated', self.timeoutActivated)

        with open('thinFlow_config.cfg', 'w') as configfile:
            self.configFile.write(configfile)

    def change_cleanup_state(self):
        """Active/Deactive cleanup"""
        self.cleanUpEnabled = (self.cleanUpCheckBox.isChecked())

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

        self.adjust_resettingSwitchButton(None)

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

        self.adjust_resettingSwitchButton(edge)

    def adjust_resettingSwitchButton(self, edge):
        """Adjustment of resettingSwitchButton in GUI"""
        if edge is None:
            # Turn button off
            self.gttr('resettingSwitchButton').setText("Off")
            self.gttr('resettingSwitchButton').setEnabled(False)
        else:
            # Turn button on, adjust Label accordingly
            resettingStatusBool = self.gttr('network')[edge[0]][edge[1]]['TFC']['resettingEnabled']
            resettingSwitchButtonLabel = "On" if resettingStatusBool else "Off"
            self.gttr('resettingSwitchButton').setText(resettingSwitchButtonLabel)
            self.gttr('resettingSwitchButton').setEnabled(True)

    def re_init_NTF_frame(self, newThinFlow=False):
        """Reinits the NTF frame"""
        if not newThinFlow:
            if self.gttr('plotNTFCanvas') is not None:
                self.gttr('plotNTFCanvas').setParent(None)
            self.sttr('plotNTFCanvas', tfType=self.currentTF, value=None)
        else:
            """Reinitialization of plotNTFCanvas with given thinflow"""
            self.gttr('plotNTFCanvas').setParent(None)
            self.sttr('plotNTFCanvas', self.currentTF,
                      PlotNTFCanvas(self.gttr('interval').network, self, intervalID=None,
                                    stretchFactor=self.plotNTFCanvasStretchFactor,
                                    showNoFlowEdges=self.showEdgesWithoutFlowCheckBox.isChecked(),
                                    onlyNTF=True))
            self.gttr('plotNTFFrameLayout').addWidget(self.gttr('plotNTFCanvas'))

    def re_init_app(self, NoNewGraph=False):
        """
        Clears the graph creation frame for new graph creation
        :param NoNewGraph: (bool) - specify whether a new graph should be initiated or the old one kept
        """
        if not NoNewGraph:
            self.sttr('network', self.currentTF, app_Interface.init_graph())  # Reinstantiation of the CurrentGraph

        # Reinitialization of graphCreationCanvas
        self.gttr('graphCreationCanvas').setParent(None)  # Drop graphCreationCanvas widget
        self.sttr('graphCreationCanvas', self.currentTF,
                  PlotCanvas(self.gttr('network'), self, self.plotCanvasStretchFactor, onlyNTF=True,
                             type=self.currentTF))
        self.gttr('plotFrameLayout').addWidget(
            self.gttr('graphCreationCanvas'))  # Add graphCreationCanvas-widget to application

        # Reinitialization of plotNTFCanvas
        self.gttr('plotNTFCanvas').setParent(None)
        self.sttr('plotNTFCanvas', self.currentTF, PlotNTFCanvas(nx.DiGraph(), self, intervalID=None,
                                                                 stretchFactor=self.plotNTFCanvasStretchFactor,
                                                                 showNoFlowEdges=self.showEdgesWithoutFlowCheckBox.isChecked(),
                                                                 onlyNTF=True))
        self.gttr('plotNTFFrameLayout').addWidget(self.gttr('plotNTFCanvas'))

        # Update UI
        self.update_node_display()
        self.update_edge_display()
        self.inflowLineEdit.setText(str(self.gttr('network').graph['inflowRate']))

        self.re_init_node_list()
        self.re_init_edge_list()

    def change_resetting(self):
        """Changes the resettingEnabled status of an edge"""
        edge = self.gttr('graphCreationCanvas').focusEdge
        if edge is None:
            return

        # Change resettingEnabled Boolean
        self.gttr('network')[edge[0]][edge[1]]['TFC']['resettingEnabled'] = not self.gttr('network')[edge[0]][edge[1]]['TFC']['resettingEnabled']
        self.adjust_resettingSwitchButton(edge)  # Change button accordingly

        # Update display
        self.gttr('graphCreationCanvas').update_edges(color=True)

    def change_no_flow_show_state(self):
        """Show/Hide edges without flow in each NTF Plot"""
        for tfType in self.tfTypeList:
            self.gttr('plotNTFCanvas', tfType).change_edge_show_status(
                show=self.showEdgesWithoutFlowCheckBox.isChecked())

    def pressed_delete(self):
        """Slot for DEL Key"""
        if self.gttr('graphCreationCanvas').focusNode is not None:
            self.delete_node()
        elif self.gttr('graphCreationCanvas').focusEdge is not None:
            self.delete_edge()

    def cleanup(self):
        """Cleanup if activated. Note: In NFC this functionality is part of the nashFlow Class"""
        if self.cleanUpEnabled:
            rmtree(self.gttr('interval').rootPath)

    def load_graph(self, graphPath=None):
        """Load graph instance from '.cg' file"""
        if not graphPath:
            dialog = QtWidgets.QFileDialog
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

        # Make sure that each edge has the property 'resettingEnabled'
        for edge in network.edges():
            v, w = edge
            try:
                property = network[v][w]['TFC']
            except KeyError:
                network[v][w]['TFC']['resettingEnabled'] = None

        if not graphPath:
            self.defaultLoadSaveDir = os.path.dirname(fopen)
            self.save_config()

        try:
            type = network.graph['type']  # Either general or spillback right now
        except KeyError:
            network.graph['type'] = 'general'

        self.sttr('network', network.graph['type'], network)
        self.currentTF = network.graph['type']
        self.tabWidget.setCurrentIndex(self.tfTypeList.index(network.graph['type']))

        self.re_init_app(NoNewGraph=True)

    def load_thinflow(self):
        """Load thinflow '.tf' file"""
        dialog = QtWidgets.QFileDialog
        fopen = dialog.getOpenFileName(self, "Select File", self.defaultLoadSaveDir, "thinflow files (*.tf)")

        fopen = fopen[0]
        if len(fopen) == 0:
            return
        fopen = str(fopen)

        # Read file
        with open(fopen, 'rb') as f:
            interval = pickle.load(f)

        tfType = 'spillback' if 'spillback' in interval.__class__.__name__ else 'general'
        self.sttr('interval', tfType, interval)
        self.currentTF = tfType
        self.tabWidget.setCurrentIndex(self.tfTypeList.index(tfType))

        self.defaultLoadSaveDir = os.path.dirname(fopen)
        self.save_config()

        self.re_init_NTF_frame(newThinFlow=True)

    def save_graph(self, graphPath=None):
        """
        Save graph instance to '.cg' file
        :param graphPath: If given, then save graph at path graphPath. Else a dialog is opened
        :return: 
        """
        self.gttr('network').graph['inflowRate'] = float(self.inflowLineEdit.text())

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

        self.gttr('network').graph['type'] = self.currentTF

        # Save network instance to file
        with open(fsave, 'wb') as f:
            pickle.dump(self.gttr('network'), f)

    def save_thinflow(self):
        """Save thinflow to '.tf' file"""
        dialog = QtWidgets.QFileDialog
        fsave = dialog.getSaveFileName(self, "Select File", self.defaultLoadSaveDir, "thinflow files (*.tf)")

        fsave = fsave[0]
        if len(fsave) == 0:
            return
        fsave = str(fsave)

        if not fsave.endswith('tf'):
            fsave += ".tf"
        self.defaultLoadSaveDir = os.path.dirname(fsave)
        self.save_config()

        # Save network instance to file
        with open(fsave, 'wb') as f:
            pickle.dump(self.gttr('interval'), f)

    def open_nfc(self, moveGraph=None):
        """
        Opens NashFlowComputation Tool
        :param moveGraph: network that should be moved, None if not specified
        :return: 
        """

        if not moveGraph:
            # Just open the application
            cmd = ['python', '../mainControl.py']
        else:
            # Open the application with a certain graph
            # Save the graph
            tmpDir = gettempdir()
            tmpFileName = Utilities.get_time()
            tmpFilePath = os.path.join(tmpDir, tmpFileName)
            self.save_graph(graphPath=tmpFilePath)
            tmpFilePathArgument = tmpFilePath + '.cg'

            cmd = ['python', '../mainControl.py', '-l', tmpFilePathArgument]

        def open_nfc_thread():
            self.proc = subprocess.Popen(cmd)
            self.proc.communicate()

        thread = threading.Thread(target=open_nfc_thread)
        thread.start()

    def move_to_nfc(self):
        self.open_nfc(moveGraph=self.gttr('network'))

    def compute_NTF(self):
        """Computes NTF in current tab"""

        network = self.gttr('network')

        # Validate input
        returnCode = self.validate_thinflow_input(network)
        if returnCode != 0:
            # Invalid input has been given
            # Spawn warning
            QtWidgets.QMessageBox.question(QtWidgets.QWidget(), 'Abort: Input error', self.get_error_message(returnCode),
                                       QtWidgets.QMessageBox.Ok)
            return

        # Drop current NTF plot
        self.re_init_NTF_frame()

        # Get necessary data
        resettingEdges = [edge for edge in network.edges() if network[edge[0]][edge[1]]['TFC']['resettingEnabled']]
        lowerBoundTime = 0  # No needed for different times as only one flowInterval is being computed
        inflowRate = float(self.inflowLineEdit.text())
        minCapacity = Utilities.compute_min_attr_of_network(network)
        counter = "Standalone"
        rootPath = self.outputDirectory

        if self.currentTF == 'general':
            templateFile = os.path.join(os.getcwd(), 'templates',
                                        'algorithm_' + str(self.templateFile + 1) + '.zpl')
        elif self.currentTF == 'spillback':
            templateFile = os.path.join(os.getcwd(), 'templates',
                                        'algorithm_spillback_' + str(self.templateFile + 1) + '.zpl')
        scipFile = self.scipFile
        timeout = float(self.timeoutLineEdit.text())

        self.save_config()

        if self.currentTF == 'general':
            self.interval_general = FlowInterval(network, resettingEdges=resettingEdges, lowerBoundTime=lowerBoundTime,
                                                 inflowRate=inflowRate, minCapacity=minCapacity, counter=counter,
                                                 outputDirectory=rootPath, templateFile=templateFile, scipFile=scipFile,
                                                 timeout=timeout)
        elif self.currentTF == 'spillback':
            fullEdges = []
            minInflowBound = float('inf')
            for e in network.edges():
                (v, w) = e
                minInflowBound = min(minInflowBound, network[v][w]['TFC']['inflowBound'])
            self.interval_spillback = FlowInterval_spillback(network, resettingEdges=resettingEdges,
                                                             fullEdges=fullEdges,lowerBoundTime=lowerBoundTime,
                                                             inflowRate=inflowRate, minCapacity=minCapacity,
                                                             counter=counter,
                                                             outputDirectory=rootPath, templateFile=templateFile,
                                                             scipFile=scipFile,
                                                             timeout=timeout, minInflowBound=minInflowBound)

        # Set shortest path network manually to entire graph (is the deepcopy really needed?)
        interval = self.gttr('interval')
        interval.shortestPathNetwork = deepcopy(network)
        self.interval_spillback.transfer_inflowBound(interval.shortestPathNetwork)

        self.advancedAlgo = (templateFile == 2)  # If true, then advanced backtracking with preprocessing is performed

        if self.currentTF == 'general':
            if self.advancedAlgo:
                interval.get_ntf_advanced()
            else:
                interval.get_ntf()
        elif self.currentTF == 'spillback':
            interval.get_ntf()

        self.sttr('plotNTFCanvas', self.currentTF, PlotNTFCanvas(interval.shortestPathNetwork, self, intervalID=None,
                                                                 stretchFactor=self.plotNTFCanvasStretchFactor,
                                                                 showNoFlowEdges=self.showEdgesWithoutFlowCheckBox.isChecked(),
                                                                 onlyNTF=True))

        self.gttr('plotNTFFrameLayout').addWidget(self.gttr('plotNTFCanvas'))
        self.cleanup()

    def validate_thinflow_input(self, network):
        """Checks whether network satisfies certain conditions, return respective error code if necessary"""
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
            try:
                m = float('inf')
                for edge in network.edges():
                    (v, w) = edge
                    m = min(m, network[v][w]['TFC']['inflowBound'])
                if m <= 0:
                    return 4
            except KeyError:
                pass

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
                    if network[v][w]['TFC']['inflowBound'] < float(
                            self.inflowLineEdit.text()):  # TODO REALLY INFLOWBOUND, NOT CAPACITY?
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
            4: "Edge capacities and inflow bounds have to be positive.",
            5: "Network contains a cycle.",
            6: "Inflow bounds of all outgoing edges from sink 's' have to be greater-or-equal than network inflow-rate."
        }

        return errorDescription[errorCode]
