import heapq
import json
import random
from typing import List

from numpy import inf

from GRF_Node import GRF_Node
from GraphAlgoInterface import GraphAlgoInterface
from DiGraph import DiGraph
from GraphInterface import GraphInterface
import matplotlib
import matplotlib.pyplot as plt

class GraphAlgo(GraphAlgoInterface):

    def __init__(self, diGraph: DiGraph = None):

        '''
        each graph-Algo object is based on DiGraph on which the algorithms are implemented
        '''
        self.Graph = diGraph

    def get_graph(self) -> GraphInterface:
        return self.Graph

    def load_from_json(self, file_name: str) -> bool:

        graf = DiGraph()  # creating an instance of DiGraph object
        with open(file_name, mode='r') as f:
            my_dictStr = json.load(f)
            for i in my_dictStr.get("Nodes"):
                if "pos" in i and len(i["pos"]) > 0:  # drag pos from within Nodes sect. block
                    my_pos_coordinates = []
                    pos_as_an_str = i.get("pos")  # for every such line do- --split the coordinates
                    my_coordinate = pos_as_an_str.split(",")
                    for j in my_coordinate:
                        my_pos_coordinates.append(j)
                    graf.add_node(i.get("id"), tuple(my_pos_coordinates))
                else:  # we are being faced with edges kind of structure
                    x = random.uniform(0, 100)
                    y = random.uniform(0, 100)
                    for k in my_coordinate:
                        graf.add_node(k.get("id"), tuple(x, y, 0))
            for j in my_dictStr.get("Edges"):
                graf.add_edge(int(j.get("src")), int(j.get("dest")), float(j.get("w")))
            self.Graph = graf
            return True
        return False

    def save_to_json(self, file_name: str) -> bool:
        try:
            with open(file_name, 'w') as my_f:
                json.dumps( file_name, indent=6, default=lambda o: o.as_dict, fp = my_f)    #fp argument represents the kind of file
                                                                                        #the json form. should be converted to
        except IOError as e:
            print(e)
            return False
        return True

    def plot_graph(self) -> None:
        graph = self.Graph
        if len(graph.Vertices)>0:
            Nodes = {}
            for point in graph.get_all_v().values():
                if point.loc is None:
                    temp = graph.v_size() / 2
                    while True:
                        rand1 = random.uniform(1, temp)
                        rand2 = random.uniform(1, temp)
                        if rand1 not in Nodes:
                            Nodes[rand1] = {}
                        if Nodes.get(rand1).get(rand2) is None:
                            Nodes[rand1][rand2] = point.id  # to notate which point is has missing values
                            break
                    point.loc = (random.uniform(1, temp), random.uniform(1, temp), 0)
                plt.plot(point.loc[0], point.loc[1], 'o')
                plt.text(point.loc[0], point.loc[1], point.id, fontsize=7, color='red')
            for v in graph.get_all_v().values():
                for j in graph.Vertices.get(v).get_connections_out():
                    p1 = v.loc[0]
                    p2 = v.loc[1]
                    pj = j.loc[0]
                    pj2 = j.loc[1]
                    plt.arrow(p1, p2, pJ - p1, pJ2 - p2, length_includes_head=True, head_width=0.0002, width=0)
        plt.show()

    def dijkstra(self, src, dest) -> (float, list):

        distances = {node: inf for node in self.Graph.Vertices.keys()}
        visited_nodes = {0: src}  # contains all the nodes from source node to the start of list
        distances[src] = 0
        queue = []  # represents nodes within tuples of (weight,node_object)
        heapq.heappush(queue, (0, src))

        while queue:  # meaning as long as the queue is not empty
            curr_node = heapq.heappop(queue)[1]

            if distances[curr_node] == inf:
                break

            for neighbours, w in self.Graph.Vertices.get(curr_node).get_connections_out().items():
                optional_dist = distances[curr_node] + w
                if optional_dist < distances[neighbours]:
                    distances[neighbours] = optional_dist
                    visited_nodes[neighbours] = curr_node
                    heapq.heappush(queue, (curr_node, optional_dist))  # in-order to maintain those shorter dists

                if distances[dest] == inf:
                    return inf, []

                path = []
                curr_node = dest

                while curr_node != -1:
                    path.insert(0, curr_node)
                    curr_node = visited_nodes[curr_node]

                return distances[dest], path  # note that we should return the equivalent distance to the end of path

    def shortest_path(self, id1: int, id2: int) -> (float, list):

        if self.Graph.Vertices.get(id1) is None:
            raise Exception("Node notated as id1 doesn't exist within the graph")

        if self.Graph.Vertices.get(id2) is None:
            raise Exception("Node notated as id2 doesn't exist within the graph")

        if id1 == id2:
            return 0, [
                id1]  # in the occasion where the starting and ending point are the same- the path passes only the starting v

        return self.dijkstra(id1, id2)

    def distsforVert(self, vert: int):
        sum = 0.0
        for v in self.Graph.Vertices:
            if v != vert:
                sum += self.shortest_path(v, vert)
        return sum

    def centerPoint(self) -> (int, float):
        if self.connected_grf():
            min = inf
            center = None
            for v in self.Graph.Vertices.items():
                upd_sum_Of_dists = self.distsforVert(v)
                if upd_sum_Of_dists < min:
                    min = upd_sum_Of_dists
                    center = v

            return center, min
        return None

    def dfs(self, graf: DiGraph, id: int, visited_NO: list):
        stack_myNode = [id]  # at first, we initialize the stack to contain only the starting node,
        while stack_myNode:  # from which the graph is being traversed
            stack_myNode.pop()
            v = stack_myNode[-1]
            if visited_NO[v]:
                v = stack_myNode.pop()
                if visited_NO[v] == 1:
                    visited_NO[v] = 2
                    stack_myNode.append(v)

            else:
                visited_NO[v] = 1
                for k in graf.all_in_edges_of_node(v):
                    if visited_NO[k] == 0:
                        if not visited_NO[k]:
                            stack_myNode.append(k)

    def connected_grf(self):
        num_Vertices = len(self.Graph.Vertices)
        for i in range(num_Vertices):
            visited = [False] * num_Vertices
            self.dfs(self.Graph, i, visited)
            # afterwards we got elements whose were visited
            for b in visited:
                if b not in visited:
                    return False
            return True

        def TSP(self, node_lst: List[int]) -> (List[int], float):
            List_traj = []
            start_N = node_lst[0]
            tsp = []
            stack = []
            tsp.append(start_N)
            for k,v in self.Graph.Vertices.items():
                for next_point in self.Graph.Vertices.get(v).all_out_edges_of_Node[0]:
                    if start_N == v and node_lst[k] == next_point:        #it simbolize if curr start from the curr iterated vertex
                        node_lst[0] = tsp[k]                              #and in addition whether the current iterated node comes to our dest
                        req_num = node_lst[k]
                        node_lst.remove(req_num)
                        stack.push(1)
                        tsp.append(node_lst)
                    if len(stack) == 0:
                        for m, vert in self.Graph.Vertices.items():
                            for point in self.Graph.Vertices.get(vert).get_connections_out:
                                if point in tsp:
                                    if tsp[len(tsp)-1] == vert and node_lst.get(0).id == point.id:
                                        node_lst[0] = tsp[k]
                                        stack.pop()
                                        List_traj.append(node_lst)

                    break

            if len(stack) == 0:
                tsp_out = []
                min = inf
                for list_rel in List_traj:
                    sum_currEdges = 0
                    first_p_traj = list[0]
                    for k,v in self.Graph.Vertices.get(first_p_traj):
                        for next_node in self.Graph.Vertices.get(v).all_out_edges_of_Node:
                            curr_weight = next_node[1]
                            if list[i-1] == v and list[i] == next_node[0]:
                                sum += curr_weight


                    if sum<min:
                        min = sum
                        tsp_out = list_rel

            return tsp








