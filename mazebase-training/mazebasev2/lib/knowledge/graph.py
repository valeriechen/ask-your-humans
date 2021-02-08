from .rules import Rule
import pdb

# A generic graph class
class Graph(object):
    def __init__(self):
        self.nodes = {}

    def add_node(self, name, data):
        node = Node(name, data)
        if name in self.nodes:
            pdb.set_trace()
            assert(name not in self.nodes)
        self.nodes[name] = node

    # Make a directed edge on the graph from node 1 to node 2
    def add_edge(self, node1_key, node2_key):
        assert(node1_key in self.nodes)
        node1 = self.nodes[node1_key]
        assert(node2_key in self.nodes)
        node2 = self.nodes[node2_key]
        node1.outgoing.append(node2)
        node2.incoming.append(node1)

class Node(object):
    def __init__(self, key, data):
        self.key = key
        self.data = data
        self.incoming = []
        self.outgoing = []

# Class that deals with hierarchy rules
# TODO - this might be unnecessary later
class HierarchyTree(Graph):
    def __init__(self, rules):
        super(HierarchyTree, self).__init__()
        # For all items with any connections, item is key
        # Value is dict with 'parents' and 'children' with list of other items

        # TODO - haven't implemented this part of knowledge
        hierarchy_tree = {}
