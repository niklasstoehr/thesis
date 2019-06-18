import random
import networkx as nx
import numpy as np

class Base_Samplers():

    def __init__(self, g_complete, a_complete):
        self.g_complete = g_complete
        self.n_complete = len(g_complete)
        self.a_complete = a_complete



    def bfs(self, source_starts, depth):

        g = nx.Graph()

        for i in range(0, source_starts):
            e = nx.bfs_edges(self.g_complete, source=random.randint(0, self.n_complete),
                             depth_limit=depth)
            g.add_edges_from(list(e))
            return g



    def walk(self, source_starts, source_return, random_p):

        ## create random walks _________
        g = nx.Graph()

        nodes = []
        edges = []

        for i in range(0, source_starts):

            start_node = random.randint(0, self.n_complete)

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
            nodes.append(random.randint(0, self.n_complete))

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
                next_node = random.randint(0, self.n_complete)
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

