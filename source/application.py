# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         application.py
# Description:  Interface class; controlling signals/slots & communication between widgets et cetera
# ===========================================================================

import ConfigParser
import os
import pickle
from warnings import filterwarnings

import networkx as nx

from nashFlowClass import NashFlow
from plotAnimationCanvasClass import PlotAnimationCanvas
from plotCanvasClass import PlotCanvas
from plotNTFCanvasClass import PlotNTFCanvas
from plotValuesCanvasClass import PlotValuesCanvas
from ui import mainWdw
from utilitiesClass import Utilities

filterwarnings('ignore')  # For the moment: ignore warnings as pyplot.hold is deprecated

if os.name == 'posix':
    from PyQt4 import QtGui, QtCore
else:
    from PySide import QtGui

TOL = 1e-8


# =======================================================================================================================


class Interface(QtGui.QMainWindow, mainWdw.Ui_MainWindow):
    """Controls GUI"""

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)


        # Scaling factors of frames
        self.plotCanvasStretchFactor = float(self.plotFrame.width())/self.plotFrame.height()
        self.plotAnimationCanvasStretchFactor = float(self.plotAnimationFrame.width())/self.plotAnimationFrame.height()
        self.plotNTFCanvasStretchFactor = float(self.plotNTFFrame.width())/self.plotNTFFrame.height()

        # Init graph
        self.network = self.init_graph()
        self.init_graph_creation_app()
        self.inflowLineEdit.setText('1')


        self.outputDirectory = ''
        self.templateFile = ''
        self.scipFile = ''

        self.defaultLoadSaveDir = ''

        self.numberOfIntervals = -1

        self.cleanUpEnabled = True

        self.animationLowerBound = 0
        self.animationUpperBound = 1

        self.configFile = ConfigParser.RawConfigParser()

        self.init_nashflow_app()

        self.load_config()  # Try to load configuration file

        self.tabWidget.setCurrentIndex(0)  # Show Graph Creation Tab

        # Signal configuration
        self.updateNodeButton.clicked.connect(self.update_node)
        self.deleteNodeButton.clicked.connect(self.delete_node)
        self.updateEdgeButton.clicked.connect(self.update_add_edge)
        self.deleteEdgeButton.clicked.connect(self.delete_edge)
        self.outputDirectoryPushButton.clicked.connect(self.select_output_directory)
        self.templateFilePushButton.clicked.connect(self.select_template_file)
        self.scipPathPushButton.clicked.connect(self.select_scip_binary)
        self.computeFlowPushButton.clicked.connect(self.compute_nash_flow)
        self.cleanUpCheckBox.clicked.connect(self.change_cleanup_state)
        self.exportDiagramPushButton.clicked.connect(self.export_diagram)
        self.setTimePushButton.clicked.connect(self.set_new_time_manually)

        # Configure Slider
        self.timeSlider.setMinimum(0)
        self.timeSlider.setMaximum(99)
        self.timeSlider.setValue(0)
        self.timeSlider.setTickPosition(2) # Set ticks below horizontal slider
        self.timeSlider.setTickInterval(1)

        self.timeSlider.valueChanged.connect(self.slider_value_change)
        self.timeSlider.sliderReleased.connect(self.slider_released)

        self.actionNew_graph.triggered.connect(self.re_init_graph_creation_app)
        self.actionLoad_graph.triggered.connect(self.load_graph)
        self.actionSave_graph.triggered.connect(self.save_graph)
        self.actionExit.triggered.connect(QtGui.QApplication.quit)
        self.actionLoad_NashFlow.triggered.connect(self.load_nashflow)
        self.actionSave_NashFlow.triggered.connect(self.save_nashflow)


        self.intervalsListWidget.itemClicked.connect(self.update_NTF_display)

        self.generateAnimationPushButton.clicked.connect(self.generate_animation)

        # Keyboard shortcuts
        QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Delete), self).activated.connect(
            self.pressed_delete)  # Pressed Delete
        # Edge shortcuts
        self.tailLineEdit.returnPressed.connect(self.update_add_edge)
        self.headLineEdit.returnPressed.connect(self.update_add_edge)
        self.capacityLineEdit.returnPressed.connect(self.update_add_edge)
        self.transitTimeLineEdit.returnPressed.connect(self.update_add_edge)
        # Node shortcuts
        self.nodeNameLineEdit.returnPressed.connect(self.update_node)
        self.nodeXLineEdit.returnPressed.connect(self.update_node)
        self.nodeYLineEdit.returnPressed.connect(self.update_node)

        # Animation generation shortcuts
        self.animationStartLineEdit.returnPressed.connect(self.generate_animation)
        self.animationEndLineEdit.returnPressed.connect(self.generate_animation)
        self.setTimeLineEdit.returnPressed.connect(self.set_new_time_manually)





    @staticmethod
    def init_graph():

        # Graph Creation
        network = nx.DiGraph()
        network.add_nodes_from(
            [('s', {'position': (-90, 0), 'label': 's'}), ('t', {'position': (90, 0), 'label': 't'})])
        network.graph['lastID'] = network.number_of_nodes() - 2  # Keep track of next nodes ID
        network.graph['inflowRate'] = 1

        return network

    def init_graph_creation_app(self):
        # Configure plotFrame to display plots of graphs
        self.plotFrameLayout = QtGui.QVBoxLayout()
        self.plotFrame.setLayout(self.plotFrameLayout)
        self.graphCreationCanvas = PlotCanvas(self.network, self)  # Initialize PlotCanvas
        self.plotFrameLayout.addWidget(self.graphCreationCanvas)

    def init_nashflow_app(self):
        # Configure plotNTFFrame to display plots of NTF
        self.plotNTFFrameLayout = QtGui.QVBoxLayout()
        self.plotNTFFrame.setLayout(self.plotNTFFrameLayout)
        self.plotNTFCanvas = None
        self.NTFPlotList = []

        # Configure plotAnimationFrame to display animation
        self.plotAnimationFrameLayout = QtGui.QVBoxLayout()
        self.plotAnimationFrame.setLayout(self.plotAnimationFrameLayout)
        self.plotAnimationCanvas = None

        # Configure plotDiagramFrame to display edge in- and outflow, queue-size and node labels
        self.plotDiagramLayout = QtGui.QVBoxLayout()
        self.plotDiagramFrame.setLayout(self.plotDiagramLayout)
        self.plotDiagramCanvas = PlotValuesCanvas(callback=self.callback_plotValuesCanvas)
        self.plotDiagramLayout.addWidget(self.plotDiagramCanvas)





    def generate_animation(self):
        lowerBoundInput = str(self.animationStartLineEdit.text())
        lowerBound = float(lowerBoundInput) if lowerBoundInput != "" else 0

        upperBoundInput = str(self.animationEndLineEdit.text())
        if upperBoundInput == "":
            upperBoundInput = self.nashFlow.flowIntervals[-1][1] \
                if self.nashFlow.flowIntervals[-1][1] < float('inf') \
                else Utilities.add_3_and_round_up(self.nashFlow.flowIntervals[-1][0])

        upperBound = float(upperBoundInput)


        self.animationLowerBound = lowerBound
        self.animationUpperBound = upperBound
        self.set_plot_range()

        self.timeSlider.setValue(0)
        self.plotAnimationCanvas.reset_bounds(self.animationLowerBound, self.animationUpperBound)



    def update_node_display(self):
        """Update display of the properties of the currently focussed node self.graphCreationCanvas.focusNode, if existing"""
        if self.graphCreationCanvas.focusNode is not None:
            # TO DO: Check for valid input
            vertex = self.graphCreationCanvas.focusNode
            self.nodeNameLineEdit.setText(self.network.node[vertex]['label'])
            self.nodeXLineEdit.setText(str(self.network.node[vertex]['position'][0]))
            self.nodeYLineEdit.setText(str(self.network.node[vertex]['position'][1]))

        else:
            self.nodeNameLineEdit.setText("")
            self.nodeXLineEdit.setText("")
            self.nodeYLineEdit.setText("")

        self.setFocus()


    def update_node(self):
        """Update attributes of focusNode"""
        if self.graphCreationCanvas.focusNode is None:
            return

        nodeName = str(self.nodeNameLineEdit.text())
        XPos = str(self.nodeXLineEdit.text())
        YPos = str(self.nodeYLineEdit.text())
        if len(nodeName) > 0 and len(XPos) > 0 and len(YPos) > 0:
            # TO DO: Check for valid input
            vertex = self.graphCreationCanvas.focusNode


            self.network.node[vertex]['label'] = nodeName
            movedBool = (self.network.node[vertex]['position'] != (int(XPos), int(YPos)))
            self.network.node[vertex]['position'] = (int(XPos), int(YPos))

            self.graphCreationCanvas.update_nodes(moved=movedBool)  # Update UI
            if movedBool:
                self.graphCreationCanvas.update_edges(moved=movedBool)





    def delete_node(self):
        """Delete focusNode from network"""
        vertex = self.graphCreationCanvas.focusNode
        if vertex is None or vertex in ['s', 't']:
            return

        if vertex in self.network:
            self.graphCreationCanvas.update_nodes(removal=True, color=True)
            numberOfEdges = self.network.number_of_edges()
            self.network.remove_node(vertex)
            removedEdgeBool = (numberOfEdges > self.network.number_of_edges())
            self.graphCreationCanvas.focusNode = None

            if removedEdgeBool:
                self.graphCreationCanvas.update_edges(removal=True)

            # Update UI
            self.update_node_display()

    def update_edge_display(self):
        """Update display of the properties of the currently focussed edge focusEdge, if existing"""
        edge = self.graphCreationCanvas.focusEdge

        if edge is not None:
            # TO DO: Check for valid input
            self.tailLineEdit.setText(self.network.node[edge[0]]['label'])
            self.headLineEdit.setText(self.network.node[edge[1]]['label'])
            self.transitTimeLineEdit.setText(
                str(self.network[edge[0]][edge[1]]['transitTime']))
            self.capacityLineEdit.setText(
                str(self.network[edge[0]][edge[1]]['capacity']))
        else:
            self.tailLineEdit.setText("")
            self.headLineEdit.setText("")
            self.transitTimeLineEdit.setText("")
            self.capacityLineEdit.setText("")

        self.setFocus()

    def update_add_edge(self):
        """Add an edge or update attributes of focusNode, if existing"""
        if self.graphCreationCanvas.focusEdge is None:
            return
        tailLabel = str(self.tailLineEdit.text())
        headLabel = str(self.headLineEdit.text())
        transitText = float(self.transitTimeLineEdit.text())
        capacityText = float(self.capacityLineEdit.text())

        # Work with actual node IDs, not labels
        labels = nx.get_node_attributes(self.network, 'label')
        tail = labels.keys()[labels.values().index(tailLabel)]
        head = labels.keys()[labels.values().index(headLabel)]

        if capacityText <= 0 or transitText < 0:
            # This is not allowed
            return

        if self.network.has_edge(tail, head):
            # Update the edges attributes
            self.network[tail][head]['transitTime'] = transitText
            self.network[tail][head]['capacity'] = capacityText
            self.graphCreationCanvas.update_edges()
        else:
            # Add a new edge
            self.network.add_edge(tail, head, transitTime=transitText, capacity=capacityText)
            self.graphCreationCanvas.focusEdge = (tail, head)
            self.graphCreationCanvas.update_edges(added=True, color=True)

        # Update UI
        self.update_edge_display()

    def delete_edge(self):
        """Delete focusEdge from network"""
        edge = self.graphCreationCanvas.focusEdge
        if edge is None:
            return

        if self.network.has_edge(edge[0], edge[1]):
            self.network.remove_edge(edge[0], edge[1])  # Deletion before update, as opposed to delete_node()
            self.graphCreationCanvas.update_edges(removal=True)

            self.graphCreationCanvas.focusEdge = None

            # Update UI
            self.update_edge_display()

    def re_init_graph_creation_app(self, NoNewGraph=False):
        """
        Clears the graph creation tab for new graph creation
        :param NoNewGraph: (bool) - specify whether a new graph should be initiated or the old one kept
        """

        if not NoNewGraph:
            self.network = self.init_graph()  # Reinstantiation of the CurrentGraph

        # Reinitialization of graphCreationCanvas
        self.graphCreationCanvas.setParent(None)  # Drop graphCreationCanvas widget
        self.graphCreationCanvas = PlotCanvas(self.network, self, self.plotCanvasStretchFactor)
        self.plotFrameLayout.addWidget(self.graphCreationCanvas)  # Add graphCreationCanvas-widget to application

        # Update UI
        self.update_node_display()
        self.update_edge_display()
        self.inflowLineEdit.setText(str(self.network.graph['inflowRate']))

    def re_init_nashflow_app(self):
        # Configure plotNTFFrame to display plots of NTF
        if self.plotNTFCanvas is not None:
            self.plotNTFCanvas.setParent(None)
        self.plotNTFCanvas = None
        self.NTFPlotList = []

        self.intervalsListWidget.clear()

        # Configure plotAnimationFrame to display animation
        if self.plotAnimationCanvas is not None:
            self.plotAnimationCanvas.setParent(None)

        self.animationUpperBound = self.nashFlow.flowIntervals[-1][1] \
                if self.nashFlow.flowIntervals[-1][1] < float('inf') \
                else Utilities.add_3_and_round_up(self.nashFlow.flowIntervals[-1][0])

        self.animationStartLineEdit.setText("%.2f" % self.animationLowerBound)
        self.animationEndLineEdit.setText("%.2f" % self.animationUpperBound)

        self.timeSlider.setValue(0)

        self.plotAnimationCanvas = PlotAnimationCanvas(nashflow=self.nashFlow, interface=self, upperBound=self.animationUpperBound, stretchFactor=self.plotAnimationCanvasStretchFactor)
        self.plotAnimationFrameLayout.addWidget(self.plotAnimationCanvas)


        # Configure plotDiagramFrame
        if self.plotDiagramCanvas is not None:
            self.plotDiagramCanvas.setParent(None)
        self.plotDiagramCanvas = PlotValuesCanvas(callback=self.callback_plotValuesCanvas)
        self.plotDiagramLayout.addWidget(self.plotDiagramCanvas)


    def load_graph(self):
        """Load CurrentGraph instance from '.cg' file"""
        dialog = QtGui.QFileDialog
        # noinspection PyCallByClass
        fopen = dialog.getOpenFileName(self, "Select File", self.defaultLoadSaveDir, "network files (*.cg)")

        if os.name != 'posix':
            fopen = fopen[0]
        if len(fopen) == 0:
            return
        fopen = str(fopen)

        # Read file         
        with open(fopen, 'rb') as f:
            self.network = pickle.load(f)
        self.defaultLoadSaveDir = os.path.dirname(fopen)
        self.save_config()
        self.re_init_graph_creation_app(NoNewGraph=True)

        self.tabWidget.setCurrentIndex(0)

    def save_graph(self):
        """Save CurrentGraph instance to '.cg' file"""

        self.network.graph['inflowRate'] = float(self.inflowLineEdit.text())

        dialog = QtGui.QFileDialog
        fsave = dialog.getSaveFileName(self, "Select File", "", "network files (*.cg)")

        if os.name != 'posix':
            fsave = fsave[0]
        if len(fsave) == 0:
            return
        fsave = str(fsave)

        if not fsave.endswith('cg'):
            fsave += ".cg"
        self.defaultLoadSaveDir = os.path.dirname(fsave)
        self.save_config()
        # Save network instance to file
        with open(fsave, 'wb') as f:
            pickle.dump(self.network, f)

    def load_nashflow(self):
        """Load NashFlow instance from '.nf' file"""
        dialog = QtGui.QFileDialog
        fopen = dialog.getOpenFileName(self, "Select File", "", "nash flow files (*.nf)")

        if os.name != 'posix':
            fopen = fopen[0]
        if len(fopen) == 0:
            return
        fopen = str(fopen)

        # Read file
        with open(fopen, 'rb') as f:
            self.nashFlow = pickle.load(f)
        self.defaultLoadSaveDir = os.path.dirname(fopen)
        self.save_config()
        self.re_init_nashflow_app()
        self.add_intervals_to_list()

        self.tabWidget.setCurrentIndex(1)
    def save_nashflow(self):
        """Save Nashflow instance to '.nf' file"""

        dialog = QtGui.QFileDialog
        fsave = dialog.getSaveFileName(self, "Select File", "", "nash flow files (*.nf)")

        if os.name != 'posix':
            fsave = fsave[0]
        if len(fsave) == 0:
            return
        fsave = str(fsave)

        if not fsave.endswith('nf'):
            fsave += ".nf"
        self.defaultLoadSaveDir = os.path.dirname(fsave)
        self.save_config()
        # Save network instance to file
        with open(fsave, 'wb') as f:
            pickle.dump(self.nashFlow, f)

    def select_output_directory(self):
        """Select output directory for nash flow computation"""
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

    def select_template_file(self):
        """Select zimpl template file"""
        defaultDir = '' if not os.path.isfile(self.templateFile) else os.path.dirname(self.templateFile)
        dialog = QtGui.QFileDialog
        fselect = dialog.getOpenFileName(self, "Select File", defaultDir, "zimpl files (*.zpl)")

        if os.name != 'posix':
            fselect = fselect[0]
        if len(fselect) == 0:
            return
        fselect = str(fselect)
        self.templateFileLineEdit.setText(fselect)
        self.templateFile = fselect


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

    def load_config(self):
        self.configFile.add_section('Settings')
        self.configFile.set('Settings', 'outputdir', '')
        self.configFile.set('Settings', 'templatefile', '')
        self.configFile.set('Settings', 'scippath', '')
        self.configFile.set('Settings', 'cleanup', '1')
        self.configFile.set('Settings', 'intervals', '-1')
        self.configFile.set('Settings', 'defaultloadsavedir', '')

        try:
            self.configFile.read('config.cfg')

            self.outputDirectory = self.configFile.get('Settings', 'outputdir')
            self.outputDirectoryLineEdit.setText(self.outputDirectory)
            self.templateFile = self.configFile.get('Settings', 'templatefile')
            self.templateFileLineEdit.setText(self.templateFile)
            self.scipFile = self.configFile.get('Settings', 'scippath')
            self.scipPathLineEdit.setText(self.scipFile)

            self.cleanUpEnabled = (self.configFile.get('Settings', 'cleanup') == 'True')
            self.cleanUpCheckBox.setChecked(self.cleanUpEnabled)

            self.numberOfIntervals = self.configFile.get('Settings', 'intervals')
            self.intervalsLineEdit.setText(self.numberOfIntervals)

            self.defaultLoadSaveDir = self.configFile.get('Settings', 'defaultloadsavedir')

        except Exception as err:
            return

    def save_config(self):
        self.configFile.set('Settings', 'outputdir', self.outputDirectory)
        self.configFile.set('Settings', 'templatefile', self.templateFile)
        self.configFile.set('Settings', 'scippath', self.scipFile)
        self.configFile.set('Settings', 'cleanup', self.cleanUpEnabled)
        self.configFile.set('Settings', 'intervals', self.numberOfIntervals)
        self.configFile.set('Settings', 'defaultloadsavedir', self.defaultLoadSaveDir)

        with open('config.cfg', 'wb') as configfile:
            self.configFile.write(configfile)

    def compute_nash_flow(self):

        # Get remaining settings
        self.numberOfIntervals = self.intervalsLineEdit.text()
        inflowRate = float(self.inflowLineEdit.text())

        self.save_config()  # Save config-settings to file
        self.tabWidget.setCurrentIndex(1)  # Switch to next tab

        self.nashFlow = NashFlow(self.network, float(inflowRate), float(self.numberOfIntervals), self.outputDirectory,
                                 self.templateFile, self.scipFile, self.cleanUpEnabled)
        self.nashFlow.run()

        self.re_init_nashflow_app()

        self.add_intervals_to_list()

        self.intervalsListWidget.setCurrentRow(0)

    def add_intervals_to_list(self):
        for index, interval in enumerate(self.nashFlow.flowIntervals):
            intervalString = 'Interval ' + str(interval[2].id) + ': [' + str("%.2f" % interval[0]) + ',' + str(
                "%.2f" % interval[1]) + '['
            item = QtGui.QListWidgetItem(intervalString)
            self.intervalsListWidget.addItem(item)  # Add item to listWidget

            plot = PlotNTFCanvas(interval[2].shortestPathNetwork, self, intervalID=index, stretchFactor=self.plotNTFCanvasStretchFactor)
            self.NTFPlotList.append(plot)  # Add NTF Plot to List

    def update_NTF_display(self):
        rowID = self.intervalsListWidget.currentRow()
        if rowID < 0:
            return

        if self.plotNTFCanvas is not None:
            self.plotNTFCanvas.setParent(None)

        self.plotNTFCanvas = self.NTFPlotList[rowID]
        self.plotNTFFrameLayout.addWidget(self.plotNTFCanvas)

        self.callback_plotValuesCanvas(self.nashFlow.flowIntervals[rowID][0], False)

    def slider_value_change(self):
        self.plotAnimationCanvas.time_changed(self.timeSlider.value())

        time = self.plotAnimationCanvas.get_time_from_tick(self.timeSlider.value())
        self.plotDiagramCanvas.change_vline_position(time)

        self.currentSliderTimeLabel.setText("%.2f" % time)


    def slider_released(self):

        time = self.plotAnimationCanvas.get_time_from_tick(self.timeSlider.value())
        # Adjust NTF if necessary
        for index, interval in enumerate(self.nashFlow.flowIntervals):
            lowerBound = interval[0]
            upperBound = interval[1]
            if lowerBound <= time <= upperBound:
                self.intervalsListWidget.setCurrentRow(index)
                if self.plotNTFCanvas is not None:
                    self.plotNTFCanvas.setParent(None)

                self.plotNTFCanvas = self.NTFPlotList[index]
                self.plotNTFFrameLayout.addWidget(self.plotNTFCanvas)


    def update_node_label_graph(self):
        if self.plotAnimationCanvas.focusNode is None:
            return

        v = self.plotAnimationCanvas.focusNode

        lowerBound = self.animationLowerBound
        upperBound = self.animationUpperBound

        xValues = [0]
        yValues = [self.nashFlow.node_label(v, 0)]
        for interval in self.nashFlow.flowIntervals:
            if interval[1] < float('inf'):
                xValues.append(interval[1])
                yValues.append(self.nashFlow.node_label(v, interval[1]))

        if upperBound > xValues[-1] and self.nashFlow.flowIntervals[-1][1] == float('inf'):
            xValues.append(upperBound)
            yValues.append(self.nashFlow.node_label(v, upperBound))

        self.plotDiagramCanvas.update_plot(lowerBound, upperBound, ["Earliest arrival time"], xValues, yValues)


    def update_edge_graphs(self):
        if self.plotAnimationCanvas.focusEdge is None:
            return
        v, w = self.plotAnimationCanvas.focusEdge[0], self.plotAnimationCanvas.focusEdge[1]

        lowerBound = self.animationLowerBound
        upperBound = self.animationUpperBound


        inflowXValues = self.nashFlow.network[v][w]['cumulativeInflow'].keys()
        inflowYValues = self.nashFlow.network[v][w]['cumulativeInflow'].values()

        if upperBound > inflowXValues[-1] and self.nashFlow.infinityReached:
            lastInflow = self.nashFlow.network[v][w]['inflow'][next(reversed(self.nashFlow.network[v][w]['inflow']))]
            val = inflowYValues[-1] + (upperBound - inflowXValues[-1]) * lastInflow
            inflowXValues.append(upperBound)
            inflowYValues.append(val)

        outflowXValues = self.nashFlow.network[v][w]['cumulativeOutflow'].keys()
        outflowYValues = self.nashFlow.network[v][w]['cumulativeOutflow'].values()

        if upperBound > outflowXValues[-1] and self.nashFlow.infinityReached:
            lastOutflow = self.nashFlow.network[v][w]['outflow'][next(reversed(self.nashFlow.network[v][w]['outflow']))]
            val = outflowYValues[-1] + (upperBound - outflowXValues[-1]) * lastOutflow
            outflowXValues.append(upperBound)
            outflowYValues.append(val)

        queueXValues = self.nashFlow.network[v][w]['queueSize'].keys()
        queueYValues = self.nashFlow.network[v][w]['queueSize'].values()

        if upperBound > queueXValues[-1] and self.nashFlow.infinityReached:
            # Queue size stays constant or grows (but queue is never empty, if not already)
            lastQueueSize = queueYValues[-1]
            lastInflowInterval = next(reversed(self.nashFlow.network[v][w]['inflow']))
            lastInflow = self.nashFlow.network[v][w]['inflow'][lastInflowInterval]

            val = max(0, lastQueueSize + (lastInflow - self.nashFlow.network[v][w]['capacity']) * (upperBound - lastInflowInterval[0]))

            queueXValues.append(upperBound)
            queueYValues.append(val)

        self.plotDiagramCanvas.update_plot(lowerBound, upperBound, ["Cumulative Inflow", "Cumulative Outflow", "Queue size"], inflowXValues, inflowYValues, (outflowXValues, outflowYValues), (queueXValues, queueYValues))

    def set_plot_range(self):
        if self.plotAnimationCanvas.focusEdge is not None:
            self.update_edge_graphs()
        elif self.plotAnimationCanvas.focusNode is not None:
            self.update_node_label_graph()

    def change_cleanup_state(self):
        self.cleanUpEnabled = (self.cleanUpCheckBox.isChecked())

    def pressed_delete(self):
        if self.tabWidget.currentIndex() != 0:
            # Deletion only possible in graph creation mode (i.e. tab 1 is focussed)
            return

        if self.graphCreationCanvas.focusNode is not None:
            self.delete_node()
        elif self.graphCreationCanvas.focusEdge is not None:
            self.delete_edge()

    def set_new_time_manually(self):
        if self.setTimeLineEdit.text() == "":
            return
        else:
            val = float(self.setTimeLineEdit.text())
            self.callback_plotValuesCanvas(xVal=val, updateNTF=True)

    def callback_plotValuesCanvas(self, xVal, updateNTF=True):
        xVal = float("%.2f" % xVal)

        valueTol = 1e-2
        if not (self.plotAnimationCanvas.timePoints[0] <= xVal <= self.plotAnimationCanvas.timePoints[-1]):
            return

        try:
            # Check if there already exists a timepoint which is sufficiently close
            xVal = next(time for time in self.plotAnimationCanvas.timePoints if Utilities.is_eq_tol(time, xVal, valueTol))
        except StopIteration:
            # Add the time point
            self.plotAnimationCanvas.add_time(xVal)
            self.timeSlider.setMaximum(self.timeSlider.maximum() + 1)

        self.timeSlider.setValue(self.plotAnimationCanvas.timePoints.index(xVal))
        if updateNTF:
            self.slider_released()

    def export_diagram(self):
        fileType = 'pdf' if self.exportComboBox.currentIndex() == 0 else 'pgf'
        dialog = QtGui.QFileDialog
        fsave = dialog.getSaveFileName(self, "Select File", self.defaultLoadSaveDir, fileType + " files (*." + fileType + ")")

        if os.name != 'posix':
            fsave = fsave[0]
        if len(fsave) == 0:
            return
        fsave = str(fsave)

        if not fsave.endswith('.' + fileType):
            fsave += '.' + fileType

        self.defaultLoadSaveDir = os.path.dirname(fsave)
        self.save_config()

        self.plotDiagramCanvas.export(path=fsave)

    def update_plotAnimationCanvas_focusSelection(self):
        if self.plotAnimationCanvas.focusNode is not None:
            self.currentFocusLineEdit.setText(str(self.plotAnimationCanvas.focusNode))
            self.currentCapacityLineEdit.setText("N/A")
            self.currentTransitTimeLineEdit.setText("N/A")
        elif self.plotAnimationCanvas.focusEdge is not None:
            v, w = self.plotAnimationCanvas.focusEdge
            self.currentFocusLineEdit.setText(str((self.network.node[v]['label'], self.network.node[w]['label'])))
            self.currentCapacityLineEdit.setText(str(self.network[v][w]['capacity']))
            self.currentTransitTimeLineEdit.setText(str(self.network[v][w]['transitTime']))
        else:
            self.currentFocusLineEdit.setText("N/A")
            self.currentCapacityLineEdit.setText("N/A")
            self.currentTransitTimeLineEdit.setText("N/A")