from hashlib import new

from GraphInterface import GraphInterface

from GRF_Node import GRF_Node


class DiGraph(GraphInterface):

    def __init__(self):
        self.Vertices = {}           #קודקודי הגרף
        self.num_of_ver = 0
        self.num_of_edges = 0
        self.mc_count= 0        #the count for number of operations performed



    def v_size(self) -> int:
       return self.num_of_ver

    def e_size(self) -> int:
        return self.num_of_edges

    def get_mc(self) -> int:
        return self.mc_count

    def add_edge(self, id1: int, id2: int, weight: float) -> bool:
        """
        Adds an edge to the graph.
        :param: id1: The start node of the edge
        :param: id2: The end node of the edge
        :param: weight: The weight of the edge (positive weight)
        :return: True if the edge was added successfully, False o.w.
        Note: If the edge already exists or one of the nodes dose not exists, the method simply does nothing
        Note2: If the weight is not positive the method raises an exception
        """
        if self.Vertices.get(id1) is None or self.Vertices.get(id2) is None:
            return False
        if id2 in self.Vertices.get(id1).get_connections_out().keys():
            return False
        if weight < 0:
            raise Exception('Edge weight must be positive')
        if id1 == id2:
            return False
        self.Vertices.get(id1).add_neighbor_out(id2, weight)
        self.Vertices.get(id2).add_neighbor_in(id1, weight)
        self.mc_count += 1
        self.num_of_edges += 1
        return True

    def all_in_edges_of_node(self, id1: int) -> dict:
        return self.Vertices.get(id1).get_connections_in()

    def add_node(self, node_id: int, pos: tuple = None) -> bool:

        if node_id not in self.Vertices.keys():
            self.Vertices[node_id] = GRF_Node(node_id, pos)
            self.num_of_ver += 1
            self.mc_count += 1  # number of operations performed on the graph
            return True

        return False

    def remove_edge(self, node_id1: int, node_id2: int) -> bool:
        if self.Vertices.get(node_id1) is None or self.Vertices.get(node_id2) is None:
            return False

        if node_id2 in self.Vertices.get(node_id1).get_connections_out():
            return False
        del self.Vertices.get(node_id1).get_connections_out()[node_id2]
        del self.Vertices.get(node_id2).get_connections_in()[node_id1]
        self.num_of_edges -= 1
        self.mc_count += 1
        return True

        return False

    def get_all_v(self) -> dict:
        return self.Vertices

    def as_dict(self):
        """
        Return the graph as dictionary {"Edges": ...., "Nodes": ....}
        :return: the graph as dictionary
        """
        node_list = []
        edge_list = []
        myO_list = []

        for k, v in self.Vertices.items():
            node_list.append(v.as_dict_node)
            for m in range(len(v.as_dict_node)):
                edge_list.append(v.as_dict_edge()[m])
            myO_list["Edges"] = edge_list
            myO_list["Nodes"] = node_list
        return myO_list

