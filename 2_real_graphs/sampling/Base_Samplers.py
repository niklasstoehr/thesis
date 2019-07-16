import random
import networkx as nx
import numpy as np


class Base_Samplers():

    def __init__(self, g_complete, a_complete):
        self.g_complete = g_complete
        self.n_complete = len(g_complete)
        self.a_complete = a_complete

    def biased_random_walk(self, n, p, q):

        g = nx.Graph()
        node_1 = random.randint(0, self.n_complete - 1)
        g.add_node(node_1)

        neigh_1 = [n for n in self.g_complete.neighbors(node_1)]
        node_2 = np.random.choice(neigh_1)
        g.add_node(node_2)
        g.add_edge(node_1, node_2)

        while len(g) < n:

            neigh_2 = [n for n in self.g_complete.neighbors(node_2)]
            neigh_2_prob = np.zeros((len(neigh_2)))

            if len(neigh_2) <= 1:  ## break the process if only neighbor is origin
                break

            for i, neigh in enumerate(neigh_2):

                if neigh == node_1:
                    neigh_2_prob[i] = 1 / p  ## go back prob -> step back

                elif g_complete.has_edge(node_1, neigh):
                    neigh_2_prob[i] = 1  ## common neighbor prob -> community exploration

                elif g_complete.has_edge(node_1, neigh) == False:
                    neigh_2_prob[i] = 1 / q  ## no common neighbor prob -> free exploration

            neigh_2_prob = neigh_2_prob / sum(neigh_2_prob)  ## normalize probability
            node_3 = np.random.choice(neigh_2, p=neigh_2_prob)
            g.add_node(node_3)
            g.add_edge(node_2, node_3)

            node_1 = node_2  ## update node 1 and node 2 for next iteration
            node_2 = node_3

        return g

    def bfs(self, random_n):

        g = nx.Graph()
        start_node = random.randint(0, self.n_complete - 1)
        g.add_node(start_node)

        queue = list()
        queue.append(start_node)

        while len(g) < random_n:

            if len(queue) > 0:
                node = queue.pop(0)
                neighbors = [n for n in self.g_complete.neighbors(node)]

                for neighbor in neighbors:

                    if g.has_edge(node, neighbor) == False and len(g) < random_n:
                        g.add_edge(node, neighbor)
                        queue.append(neighbor)

            else:
                break

        return g

    def standard_bfs(self, source_starts, depth):

        g = nx.Graph()

        for i in range(0, source_starts):
            e = nx.bfs_edges(self.g_complete, source=random.randint(0, self.n_complete - 1),
                             depth_limit=depth)
            g.add_edges_from(list(e))
            return g

    def walk(self, source_starts, source_return, random_p):

        ## create random walks _________
        g = nx.Graph()

        nodes = []
        edges = []

        for i in range(0, source_starts):

            start_node = random.randint(0, self.n_complete - 1)

            for j in range(0, source_return):

                nodes.append(start_node)

                while random_p <= np.random.rand(1):
                    # Extract vertex neighbours vertex neighborhood
                    neighbors = [n for n in self.g_complete.neighbors(nodes[-1])]
                    # Set probability of going to a neighbour is uniform
                    prob = []
                    prob = prob + [1. / len(neighbors)] * len(neighbors)
                    # Choose a vertex from the vertex neighborhood to start the next random walk
                    next_node = np.random.choice(neighbors, p=prob)

                    # Append to path
                    edges.append((nodes[-1], next_node))
                    nodes.append(next_node)

        e = set(edges)
        n = set(nodes)
        g.add_edges_from(list(e))
        g.add_nodes_from(list(n))
        return g

    ## random node selection
    def select(self, random_n):

        g = nx.Graph()

        nodes = []

        for i in range(0, random_n):
            nodes.append(random.randint(0, self.n_complete - 1))

        n = list(set(nodes))
        g.add_nodes_from(list(n))

        # e_complete = list(g_complete.edges())

        for n1 in range(0, len(n) - 1):
            for n2 in range(n1 + 1, len(n)):
                if self.g_complete.has_edge(n[n1], n[n2]):
                    self.g.add_edge(n[n1], n[n2])

        return g

    def jump(self, source_starts, random_p, jump_bias):

        ## create jump random walks _________
        g = nx.Graph()

        degrees_complete = [d for n, d in self.g_complete.degree()]
        degree_probs = np.asarray(degrees_complete) / sum(degrees_complete)

        nodes = []
        edges = []

        for i in range(0, source_starts):

            if jump_bias == "uniform":
                next_node = random.randint(0, self.n_complete - 1)
            if jump_bias == "degree":  # toDo
                next_node = np.random.choice(self.g_complete.nodes, 1, p=degree_probs)

            nodes.append(int(next_node))

            while random_p <= np.random.rand(1):
                # Extract vertex neighbours vertex neighborhood
                neighbors = [n for n in self.g_complete.neighbors(nodes[-1])]
                # Set probability of going to a neighbour is uniform
                prob = [1. / len(neighbors)] * len(neighbors)
                # Choose a vertex from the vertex neighborhood to start the next random walk
                next_node = np.random.choice(neighbors, p=prob)

                # Append to path
                edges.append((nodes[-1], next_node))
                nodes.append(next_node)

        e = set(edges)
        n = set(nodes)
        g.add_edges_from(list(e))
        g.add_nodes_from(list(n))
        return g

    ## ToDO: cut graph parts from adjacency matrix, --> potentially sort adjacency matrix

    def adjacency(self, random_n):

        g = nx.Graph()

        if random_n <= self.n_complete:
            width = random_n
        else:
            width = self.n_complete

        x = random.randint(0, self.n_complete - width)
        y = random.randint(0, self.n_complete - width)

        a_complete = (self.a_complete.toarray())
        np.random.shuffle(a_complete)  ## in-place shuffling
        a_sample = a_complete[x:x + width, y:y + width]
        g = nx.from_numpy_matrix(a_sample)
        return g


