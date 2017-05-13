# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         application.py
# Description:  Interface class; controlling signals/slots & communication between widgets et cetera
# ===========================================================================

import matplotlib
import os
import pickle
from warnings import filterwarnings

from currentGraphClass import CurrentGraph
from plotCanvasClass import PlotCanvas
from ui import mainWdw

filterwarnings('ignore')  # For the moment: ignore warnings as pyplot.hold is deprecated

if os.name == 'posix':
    from PyQt4 import QtGui
else:
    from PySide import QtGui


# =======================================================================================================================


class Interface(QtGui.QMainWindow, mainWdw.Ui_MainWindow):
    """Controls GUI"""

    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        self.setupUi(self)

        self.currentGraph = CurrentGraph()  # CurrentGraph instantiation for graph creation

        # Configure plotWidget to display plots of graphs
        self.layout = QtGui.QVBoxLayout()
        self.plotWidget.setLayout(self.layout)
        self.figure = matplotlib.figure.Figure()
        self.canvas = PlotCanvas(self.figure, self.currentGraph, self)  # Initialize PlotCanvas
        self.layout.addWidget(self.canvas)

        self.canvas.update_plot()  # Plot for the first time

        # Signal configuration
        self.updateNodeButton.clicked.connect(self.update_node)
        self.deleteNodeButton.clicked.connect(self.delete_node)
        self.updateEdgeButton.clicked.connect(self.update_add_edge)
        self.deleteEdgeButton.clicked.connect(self.delete_edge)
        self.actionNew_graph.triggered.connect(self.clear_application)
        self.actionLoad_graph.triggered.connect(self.load_graph)
        self.actionSave_graph.triggered.connect(self.save_graph)
        self.actionExit.triggered.connect(QtGui.QApplication.quit)

    def update_node_display(self):
        """Update display of the properties of the currently focussed node self.canvas.focusNode, if existing"""
        if self.canvas.focusNode is not None:
            # TO DO: Check for valid input
            self.nodeNameLineEdit.setText(self.currentGraph.label[self.canvas.focusNode])
            self.nodeXLineEdit.setText(str(self.currentGraph.position[self.canvas.focusNode][0]))
            self.nodeYLineEdit.setText(str(self.currentGraph.position[self.canvas.focusNode][1]))
        else:
            self.nodeNameLineEdit.setText("")
            self.nodeXLineEdit.setText("")
            self.nodeYLineEdit.setText("")

    def update_node(self):
        """Update attributes of focusNode"""
        nodeName = str(self.nodeNameLineEdit.text())
        XPos = str(self.nodeXLineEdit.text())
        YPos = str(self.nodeYLineEdit.text())
        if len(nodeName) > 0 and len(XPos) > 0 and len(YPos) > 0:
            # TO DO: Check for valid input
            self.currentGraph.label[self.canvas.focusNode] = nodeName
            self.currentGraph.position[self.canvas.focusNode] = (int(XPos), int(YPos))

            self.canvas.update_plot()  # Update UI

    def delete_node(self):
        """Delete focusNode from currentGraph"""
        if self.canvas.focusNode is None or self.canvas.focusNode in ['s', 't']:
            return

        if self.canvas.focusNode in self.currentGraph:
            self.currentGraph.remove_node(self.canvas.focusNode)
            del self.currentGraph.position[self.canvas.focusNode]  # for later: override within class
            del self.currentGraph.label[self.canvas.focusNode]  # for later: override within class
            self.canvas.focusNode = None

            # Update UI
            self.update_node_display()
            self.canvas.update_plot()

    def update_edge_display(self):
        """Update display of the properties of the currently focussed edge focusEdge, if existing"""
        if self.canvas.focusEdge is not None:
            # TO DO: Check for valid input
            self.tailLineEdit.setText(self.currentGraph.label[self.canvas.focusEdge[0]])
            self.headLineEdit.setText(self.currentGraph.label[self.canvas.focusEdge[1]])
            self.transitTimeLineEdit.setText(
                str(self.currentGraph[self.canvas.focusEdge[0]][self.canvas.focusEdge[1]]['transitTime']))
            self.capacityLineEdit.setText(
                str(self.currentGraph[self.canvas.focusEdge[0]][self.canvas.focusEdge[1]]['capacity']))
        else:
            self.tailLineEdit.setText("")
            self.headLineEdit.setText("")
            self.transitTimeLineEdit.setText("")
            self.capacityLineEdit.setText("")

    def update_add_edge(self):
        """Add an edge or update attributes of focusNode, if existing"""
        tailLabel = str(self.tailLineEdit.text())
        headLabel = str(self.headLineEdit.text())
        transitText = float(self.transitTimeLineEdit.text())
        capacityText = float(self.capacityLineEdit.text())

        # Work with actual node IDs, not labels
        tail = self.currentGraph.label.keys()[self.currentGraph.label.values().index(tailLabel)]
        head = self.currentGraph.label.keys()[self.currentGraph.label.values().index(headLabel)]

        if self.currentGraph.has_edge(tail, head):
            # Update the edges attributes
            self.currentGraph[tail][head]['transitTime'] = transitText
            self.currentGraph[tail][head]['capacity'] = capacityText
        else:
            # Add a new edge
            self.currentGraph.add_edge(tail, head, transitTime=transitText, capacity=capacityText)
            self.canvas.focusEdge = (tail, head)

        # Update UI
        self.canvas.update_plot()
        self.update_edge_display()

    def delete_edge(self):
        """Delete focusEdge from currentGraph"""

        if self.canvas.focusEdge is None:
            return

        if self.currentGraph.has_edge(self.canvas.focusEdge[0], self.canvas.focusEdge[1]):
            self.currentGraph.remove_edge(self.canvas.focusEdge[0], self.canvas.focusEdge[1])
            self.canvas.focusEdge = None

            # Update UI
            self.update_edge_display()
            self.canvas.update_plot()

    def clear_application(self, NoNewGraph=False):
        """
        Clears the graph creation tab for new graph creation
        :param NoNewGraph: (bool) - specify whether a new graph should be initiated or the old one kept
        """
        if not NoNewGraph:
            self.currentGraph = CurrentGraph()  # Reinstantiation of the CurrentGraph

        # Reinitialization of canvas
        self.canvas.setParent(None)  # Drop canvas widget
        self.canvas = PlotCanvas(self.figure, self.currentGraph, self)
        self.layout.addWidget(self.canvas)  # Add canvas-widget to application

        # Update UI
        self.canvas.update_plot()
        self.update_node_display()
        self.update_edge_display()

    def load_graph(self):
        """Load CurrentGraph instance from '.cg' file"""
        dialog = QtGui.QFileDialog
        fopen = dialog.getOpenFileName(self, "Select File", "", "currentGraph files (*.cg)")

        if os.name != 'posix':
            fopen = fopen[0]
        if len(fopen) == 0:
            return
        fopen = str(fopen)

        # Read file         
        with open(fopen, 'rb') as f:
            self.currentGraph = pickle.load(f)

        self.clear_application(NoNewGraph=True)

    def save_graph(self):
        """Save CurrentGraph instance to '.cg' file"""
        dialog = QtGui.QFileDialog
        fsave = dialog.getSaveFileName(self, "Select File", "", "currentGraph files (*.cg)")

        if os.name != 'posix':
            fsave = fsave[0]
        if len(fsave) == 0:
            return
        fsave = str(fsave)

        if not fsave.endswith('cg'):
            fsave += ".cg"

        # Save currentGraph instance to file
        with open(fsave, 'wb') as f:
            pickle.dump(self.currentGraph, f)
