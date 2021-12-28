class GRF_Node:
    def __init__(self, id, node_loc: tuple):
        self.edge_O = {}
        self.edge_IN = {}
        self.id = id
        self.loc = node_loc

    def add_neighbor_out(self, neighbor_id: int, weight: float) -> None:
        """
        Add "edge" that connected from this node (node_id ---> neighbor_id).
        :param neighbor_id: dest node key
        :param weight: edge's weight
        """
        self.edge_O[neighbor_id] = weight

    def add_neighbor_in(self, neighbor_id: int, weight: float) -> None:
        """
        Add "edge" that connected to this node (neighbor_id ---> node_id).
        :param neighbor_id: dest node key
        :param weight: edge's weight
        """
        self.edge_IN[neighbor_id] = weight

    def get_connections_out(self) -> dict:
        """
        Return a dictionary that holds all the "edges" that connected from this node,
        each edge is represented using a pair (key, edge weight).
        :return: dictionary (key, edge weight).
        """
        return self.edge_O

    def get_connections_in(self) -> dict:
        """
        Return a dictionary that holds all the "edges" that connected to this node,
        each edge is represented using a pair (key, edge weight).
        :return: dictionary (key, edge weight).
        """
        return self.edge_IN

    def get_key(self) -> int:
        """
        Return this node key.
        :return: key
        """
        return self.id

    def set_loc(self, _loc:tuple=None) -> None:
        self.loc = _loc


    def get_loc(self):
        return self.loc


    def as_dict_node(self) -> dict:
        loc_rep_instr = str(self.get_loc(self))
        mydict = {"coordinate": self.loc_rep_instr[1:-1] , "id": self.get_key(self)}
        return mydict

    def as_dict_edge(self) -> dict:
        edge_list = []
        for k, v in self.Vertices.get_connections_out():
            my_returned = {"src": int(self.get_key) , "w": float(v) , "dest": int(k)}
            edge_list.append(my_returned)
        return edge_list












