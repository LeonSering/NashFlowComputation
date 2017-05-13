# ===========================================================================
# Author:       Max ZIMMER
# Project:      NashFlowComputation 2017
# File:         currentGraphClass.py
# Description:  Class which inherits and governs networkx.DiGraph, allowing graph-class specific initialization
# ===========================================================================

import networkx as nx


# =======================================================================================================================

class CurrentGraph(nx.DiGraph):
    """
    Inherits and governs networkx.DiGraph class, allowing graph-class specific initialization
    """

    def __init__(self):
        super(CurrentGraph, self).__init__()  # Call parents constructor

        # Add source and sink
        self.add_nodes_from(['s', 't'])
        self.add_edge('s', 't', transitTime=50, capacity=200)

        self.lastID = self.number_of_nodes() - 2  # Keep track of next nodes ID

        # Dictionary maintains absolute position of nodes for upcoming plots
        # Ensures that (e.g. after changing the graph slightly) the plot doesn't change much 
        self.position = {'s': (-90, 0), 't': (90, 0)}

        self.label = {'s': 's', 't': 't'}  # Labels for node; IDs do not change(i.e. 's','t','1','2','3'...)
