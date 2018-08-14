# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2018
# File:         thinFlow_application.py
# Description:  Interface class; controlling signals/slots & communication between widgets
# ===========================================================================


from PyQt4 import QtGui, QtCore
import ConfigParser
import os
import pickle
import threading
import time
from shutil import rmtree
from copy import deepcopy
import networkx as nx
from warnings import filterwarnings

from plotCanvasClass import PlotCanvas
from plotNTFCanvasClass import PlotNTFCanvas
from ui import thinFlow_mainWdw
from utilitiesClass import Utilities
from application import Interface as app_Interface
from flowIntervalClass import FlowInterval



# =======================================================================================================================
filterwarnings('ignore')  # For the moment: ignore warnings as pyplot.hold is deprecated
QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)  # This is necessary if threads access the GUI


class Interface(QtGui.QMainWindow, thinFlow_mainWdw.Ui_MainWindow):
    """Controls GUI"""

    def __init__(self):
        """Initialization of Class and GUI"""
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)

        # Scaling factors of frames, to avoid distortion
        self.plotCanvasStretchFactor = float(self.plotFrame_general.width()) / self.plotFrame_general.height()
        self.plotNTFCanvasStretchFactor = float(self.plotNTFFrame_general.width()) / self.plotNTFFrame_general.height()

        # Init graph
        self.network_general = app_Interface.init_graph()
        self.init_app()
        self.inflowLineEdit.setText('1')  # Default value

        # Config defaults
        self.outputDirectory = ''
        self.templateFile = 0  # 0,1,2 for three algorithms from thesis
        self.scipFile = ''
        self.timeoutActivated = False
        self.defaultLoadSaveDir = ''
        self.cleanUpEnabled = True

        self.configFile = ConfigParser.RawConfigParser()  # This is the parser, not to confuse with the actual config.txt File, which cannot be specified

        # Initializations
        self.load_config()  # Try to load configuration file
        #self.tabWidget.setCurrentIndex(0)  # Show Graph Creation Tab # Not necessarily useful (could be done over config or startup)

        # Signal configuration
        self.updateNodeButton_general.clicked.connect(self.update_node)
        self.deleteNodeButton_general.clicked.connect(self.delete_node)
        self.updateEdgeButton_general.clicked.connect(self.update_add_edge)
        self.deleteEdgeButton_general.clicked.connect(self.delete_edge)
        self.outputDirectoryPushButton.clicked.connect(self.select_output_directory)
        self.scipPathPushButton.clicked.connect(self.select_scip_binary)
        self.cleanUpCheckBox.clicked.connect(self.change_cleanup_state)
        self.showEdgesWithoutFlowCheckBox.clicked.connect(self.change_no_flow_show_state)
        self.nodeSelectionListWidget_general.itemClicked.connect(self.update_focus_node)
        self.edgeSelectionListWidget_general.itemClicked.connect(self.update_focus_edge)
        self.activateTimeoutCheckBox.clicked.connect(self.change_timeout_state)
        self.computeThinFlowPushButton_general.clicked.connect(self.compute_NTF)
        self.resettingSwitchButton_general.clicked.connect(self.change_resetting)
        # TO BE DONE LATER
        #self.actionNew_graph.triggered.connect(self.re_init_graph_creation_app)
        #self.actionLoad_graph.triggered.connect(self.load_graph)
        #self.actionSave_graph.triggered.connect(self.save_graph)
        #self.actionExit.triggered.connect(QtGui.QApplication.quit)
        #self.actionOpen_manual.triggered.connect(self.show_help)

        # Keyboard shortcuts
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self).activated.connect(
            self.pressed_delete)  # Pressed Delete
        self.tailLineEdit_general.returnPressed.connect(self.update_add_edge)
        self.headLineEdit_general.returnPressed.connect(self.update_add_edge)
        self.capacityLineEdit_general.returnPressed.connect(self.update_add_edge)
        self.nodeNameLineEdit_general.returnPressed.connect(self.update_node)
        self.nodeXLineEdit_general.returnPressed.connect(self.update_node)
        self.nodeYLineEdit_general.returnPressed.connect(self.update_node)


    def init_app(self): # former: init_graph_creation_app
        """Initialization of Tab 0"""
        # Configure plotFrame to display plots of graphs
        self.plotFrameLayout_general = QtGui.QVBoxLayout()
        self.plotFrame_general.setLayout(self.plotFrameLayout_general)
        self.graphCreationCanvas_general = PlotCanvas(self.network_general, self, stretchFactor=1.57, onlyNTF=True)  # Initialize PlotCanvas
        self.plotFrameLayout_general.addWidget(self.graphCreationCanvas_general)

        self.re_init_node_list()
        self.re_init_edge_list()

        # Configure plotNTFFrame
        self.plotNTFFrameLayout_general = QtGui.QVBoxLayout()
        self.plotNTFFrame_general.setLayout(self.plotNTFFrameLayout_general)
        # Add empty graph to plotNTFCanvas to not destroy layout
        self.plotNTFCanvas_general = PlotNTFCanvas(nx.DiGraph(), self, intervalID=None,
                             stretchFactor=self.plotNTFCanvasStretchFactor,
                             showNoFlowEdges=self.showEdgesWithoutFlowCheckBox.isChecked(), onlyNTF=True)
        self.plotNTFFrameLayout_general.addWidget(self.plotNTFCanvas_general)

    def re_init_node_list(self):
        """Clear and fill node list"""
        self.nodeSelectionListWidget_general.clear()
        self.nodeToListItem_general = dict()
        for node in self.network_general.nodes():
            self.add_node_to_list(node)

        self.nodeSelectionListWidget_general.sortItems()

    def re_init_edge_list(self):
        """Clear and fill edge list"""
        self.edgeSelectionListWidget_general.clear()
        self.edgeToListItem_general = dict()
        for edge in self.network_general.edges():
            self.add_edge_to_list(edge)

        self.edgeSelectionListWidget_general.sortItems()

    def add_node_to_list(self, node):
        """
        Add single node to list
        :param node: node which will be added to ListWidget
        """
        nodeString = 'Node ' + str(node) + ': ' + self.network_general.node[node]['label']
        item = QtGui.QListWidgetItem(nodeString)
        self.nodeToListItem_general[node] = item
        self.nodeSelectionListWidget_general.addItem(item)  # Add item to listWidget

    def add_edge_to_list(self, edge):
        """
        Add single edge to list
        :param edge: edge which will be added to ListWidget
        """
        v, w = edge
        edgeString = 'Edge: ' + str((self.network_general.node[v]['label'], self.network_general.node[w]['label']))
        item = QtGui.QListWidgetItem(edgeString)
        self.edgeToListItem_general[edge] = item
        self.edgeSelectionListWidget_general.addItem(item)  # Add item to listWidget

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
        if self.graphCreationCanvas_general.focusNode is None:
            return

        nodeName = str(self.nodeNameLineEdit_general.text())
        XPos = str(self.nodeXLineEdit_general.text())
        YPos = str(self.nodeYLineEdit_general.text())
        if len(nodeName) > 0 and len(XPos) > 0 and len(YPos) > 0:
            vertex = self.graphCreationCanvas_general.focusNode
            if nodeName != self.network_general.node[vertex]['label']:
                self.network_general.node[vertex]['label'] = nodeName
                item = self.nodeToListItem_general[vertex]
                self.nodeSelectionListWidget_general.takeItem(self.nodeSelectionListWidget_general.row(item))  # Delete item
                self.add_node_to_list(vertex)
                self.nodeSelectionListWidget_general.sortItems()  # Re-sort

            movedBool = (self.network_general.node[vertex]['position'] != (float(XPos), float(YPos)))
            self.network_general.node[vertex]['position'] = (float(XPos), float(YPos))

            self.graphCreationCanvas_general.update_nodes(moved=movedBool, color=True)  # Update UI
            if movedBool:
                self.graphCreationCanvas_general.update_edges(moved=movedBool)

    def delete_node(self):
        """Delete focusNode from network"""
        vertex = self.graphCreationCanvas_general.focusNode
        if vertex is None or vertex in ['s', 't']:
            return

        if vertex in self.network_general:
            item = self.nodeToListItem_general[vertex]
            index = self.nodeSelectionListWidget_general.row(item)
            self.nodeSelectionListWidget_general.takeItem(index)

            for edge in self.network_general.edges():
                if vertex in edge:
                    item = self.edgeToListItem_general[edge]
                    index = self.edgeSelectionListWidget_general.row(item)
                    self.edgeSelectionListWidget_general.takeItem(index)

            self.graphCreationCanvas_general.update_nodes(removal=True, color=True)
            numberOfEdges = self.network_general.number_of_edges()
            self.network_general.remove_node(vertex)

            removedEdgeBool = (numberOfEdges > self.network_general.number_of_edges())
            self.graphCreationCanvas_general.focusNode = None

            if removedEdgeBool:
                self.graphCreationCanvas_general.update_edges(removal=True)

            self.update_node_display()  # Update UI

    def update_node_display(self):
        """Update display of the properties of the currently focussed node self.graphCreationCanvas.focusNode, if existing"""
        if self.graphCreationCanvas_general.focusNode is not None:
            vertex = self.graphCreationCanvas_general.focusNode
            self.nodeNameLineEdit_general.setText(self.network_general.node[vertex]['label'])
            self.nodeXLineEdit_general.setText(str(self.network_general.node[vertex]['position'][0]))
            self.nodeYLineEdit_general.setText(str(self.network_general.node[vertex]['position'][1]))
        else:
            self.nodeNameLineEdit_general.setText("")
            self.nodeXLineEdit_general.setText("")
            self.nodeYLineEdit_general.setText("")

        self.setFocus()  # Focus has to leave LineEdits

    def update_edge_display(self):
        """Update display of the properties of the currently focussed edge focusEdge, if existing"""
        edge = self.graphCreationCanvas_general.focusEdge
        if edge is not None:
            self.tailLineEdit_general.setText(self.network_general.node[edge[0]]['label'])
            self.headLineEdit_general.setText(self.network_general.node[edge[1]]['label'])
            self.capacityLineEdit_general.setText(
                str(self.network_general[edge[0]][edge[1]]['capacity']))
        else:
            self.tailLineEdit_general.setText("")
            self.headLineEdit_general.setText("")
            self.capacityLineEdit_general.setText("")

        self.setFocus()  # Focus has to leave LineEdits

        self.adjust_resettingSwitchButton(edge)

    def update_add_edge(self):
        """Add an edge or update attributes of focusNode, if existing"""
        if self.graphCreationCanvas_general.focusEdge is None:
            return
        tailLabel = str(self.tailLineEdit_general.text())
        headLabel = str(self.headLineEdit_general.text())
        capacityText = float(self.capacityLineEdit_general.text())

        # Work with actual node IDs, not labels
        labels = nx.get_node_attributes(self.network_general, 'label')
        tail = labels.keys()[labels.values().index(tailLabel)]
        head = labels.keys()[labels.values().index(headLabel)]

        if capacityText <= 0:
            # This is not allowed
            return

        if self.network_general.has_edge(tail, head):
            # Update the edges attributes
            self.network_general[tail][head]['capacity'] = capacityText
            self.graphCreationCanvas_general.update_edges()
        else:
            # Add a new edge
            self.network_general.add_edge(tail, head, capacity=capacityText, resettingEnabled=False)
            self.graphCreationCanvas_general.focusEdge = (tail, head)
            self.graphCreationCanvas_general.update_edges(added=True, color=True)
            self.graphCreationCanvas_general.update_nodes(color=True)
            self.add_edge_to_list((tail, head))
            self.edgeSelectionListWidget_general.sortItems()

        self.update_edge_display()  # Update UI

    def delete_edge(self):
        """Delete focusEdge from network"""
        edge = self.graphCreationCanvas_general.focusEdge
        if edge is None:
            return

        if self.network_general.has_edge(edge[0], edge[1]):
            item = self.edgeToListItem_general[edge]
            index = self.edgeSelectionListWidget_general.row(item)
            self.edgeSelectionListWidget_general.takeItem(index)

            self.network_general.remove_edge(edge[0], edge[1])  # Deletion before update, as opposed to delete_node()
            self.graphCreationCanvas_general.update_edges(removal=True)

            self.graphCreationCanvas_general.focusEdge = None

            self.update_edge_display()  # Update UI

    def select_output_directory(self):
        """Select output directory for thin flow computation"""
        defaultDir = self.outputDirectory
        dialog = QtGui.QFileDialog
        fselect = dialog.getExistingDirectory(self, "Select Directory", defaultDir)

        if os.name != 'posix':
            fselect = fselect[0]
        if len(fselect) == 0:
            return
        fselect = str(fselect)
        self.outputDirectoryLineEdit.setText(fselect)
        self.outputDirectory = fselect

    def select_scip_binary(self):
        """Select scip binary"""
        defaultDir = '' if not os.path.isfile(self.scipFile) else os.path.dirname(self.scipFile)
        dialog = QtGui.QFileDialog
        fselect = dialog.getOpenFileName(self, "Select File", defaultDir)

        if os.name != 'posix':
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

        with open('thinFlow_config.cfg', 'wb') as configfile:
            self.configFile.write(configfile)

    def change_cleanup_state(self):
        """Active/Deactive cleanup"""
        self.cleanUpEnabled = (self.cleanUpCheckBox.isChecked())

    def update_focus_node(self):
        """Select new focusNode"""
        self.graphCreationCanvas_general.focusEdge = None
        index = self.nodeSelectionListWidget_general.currentRow()
        item = self.nodeSelectionListWidget_general.item(index)
        node = self.nodeToListItem_general.keys()[self.nodeToListItem_general.values().index(item)]
        self.graphCreationCanvas_general.focusNode = node
        self.graphCreationCanvas_general.update_nodes(color=True)
        self.graphCreationCanvas_general.update_edges(color=True)
        self.update_node_display()
        self.update_edge_display()

        self.adjust_resettingSwitchButton(None)

    def update_focus_edge(self):
        """Select new focusEdge"""
        self.graphCreationCanvas_general.focusNode = None
        index = self.edgeSelectionListWidget_general.currentRow()
        item = self.edgeSelectionListWidget_general.item(index)
        edge = self.edgeToListItem_general.keys()[self.edgeToListItem_general.values().index(item)]
        self.graphCreationCanvas_general.focusEdge = edge
        self.graphCreationCanvas_general.update_nodes(color=True)
        self.graphCreationCanvas_general.update_edges(color=True)
        self.update_node_display()
        self.update_edge_display()

        self.adjust_resettingSwitchButton(edge)

    def adjust_resettingSwitchButton(self, edge):
        "Adjustment of resettingSwitchButton in GUI"
        if edge is None:
            # Turn button off
            self.resettingSwitchButton_general.setText("Off")
            self.resettingSwitchButton_general.setEnabled(False)
        else:
            # Turn button on, adjust Label accordingly
            resettingStatusBool = self.network_general[edge[0]][edge[1]]['resettingEnabled']
            resettingSwitchButtonLabel = "On" if resettingStatusBool else "Off"
            self.resettingSwitchButton_general.setText(resettingSwitchButtonLabel)
            self.resettingSwitchButton_general.setEnabled(True)

    def re_init_NTF_frame(self):
        """Clears the NTF frame"""
        if self.plotNTFCanvas_general is not None:
            self.plotNTFCanvas_general.setParent(None)
        self.plotNTFCanvas_general = None

    def change_resetting(self):
        """Changes the resettingEnabled status of an edge"""
        edge = self.graphCreationCanvas_general.focusEdge
        if edge is None:
            return

        # Change resettingEnabled Boolean
        self.network_general[edge[0]][edge[1]]['resettingEnabled'] = not self.network_general[edge[0]][edge[1]]['resettingEnabled']
        self.adjust_resettingSwitchButton(edge) # Change button accordingly

        # Update display
        self.graphCreationCanvas_general.update_edges(color=True)

    def change_no_flow_show_state(self):
        """Show/Hide edges without flow in each NTF Plot"""
        self.plotNTFCanvas_general.change_edge_show_status(show=self.showEdgesWithoutFlowCheckBox.isChecked())

    def pressed_delete(self):
        """Slot for DEL Key"""
        if self.graphCreationCanvas_general.focusNode is not None:
            self.delete_node()
        elif self.graphCreationCanvas_general.focusEdge is not None:
            self.delete_edge()

    def cleanup(self):
        """Cleanup if activated. Note: In NFC this functionality is part of the nashFlow Class"""
        if self.cleanUpEnabled:
            rmtree(self.interval_general.rootPath)


    def compute_NTF(self):
        """Computes NTF in current tab"""

        # Drop current NTF plot
        self.re_init_NTF_frame()

        # Get necessary data
        resettingEdges = [edge for edge in self.network_general.edges() if self.network_general[edge[0]][edge[1]]['resettingEnabled']]
        lowerBoundTime = 0  # No needed for different times as only one flowInterval is being computed
        inflowRate = float(self.inflowLineEdit.text())
        minCapacity = Utilities.compute_min_capacity(self.network_general)
        counter = "Standalone"
        rootPath = self.outputDirectory
        templateFile = os.path.join(os.getcwd(), 'templates',
                                         'algorithm_' + str(self.templateFile + 1) + '.zpl')
        scipFile = self.scipFile
        timeout = float(self.timeoutLineEdit.text())

        self.save_config()

        self.interval_general = FlowInterval(self.network_general, resettingEdges=resettingEdges, lowerBoundTime=lowerBoundTime,
                                inflowRate=inflowRate, minCapacity=minCapacity, counter=counter,
                                outputDirectory=rootPath, templateFile=templateFile, scipFile=scipFile,
                                timeout=timeout, )

        # Set shortest path network manually to entire graph (is the deepcopy really needed?)
        self.interval_general.shortestPathNetwork = deepcopy(self.network_general)

        self.advancedAlgo = (templateFile == 2)  # If true, then advanced backtracking with preprocessing is performed

        if self.advancedAlgo:
            self.interval_general.get_ntf_advanced()
        else:
            self.interval_general.get_ntf()

        self.plotNTFCanvas_general = PlotNTFCanvas(self.interval_general.shortestPathNetwork, self, intervalID=None,
                             stretchFactor=self.plotNTFCanvasStretchFactor,
                             showNoFlowEdges=self.showEdgesWithoutFlowCheckBox.isChecked(), onlyNTF=True)

        self.plotNTFFrameLayout_general.addWidget(self.plotNTFCanvas_general)
        self.cleanup()

