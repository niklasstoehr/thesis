import random
import time
import networkx as nx
import matplotlib.pyplot as plt

class Metropolis_Hastings():

    def __init__(self,g_complete, a_complete):
        self.G1 = nx.Graph()
        self.g_complete = g_complete
        self.a_complete = a_complete

    def mhrw(self,node,size):
        dictt = {}
        node_list = set()
        node_list.add(node)
        parent_node = node_list.pop()
        dictt[parent_node] = parent_node
        degree_p = self.g_complete.degree(parent_node)
        related_list = list(self.g_complete.neighbors(parent_node))
        node_list.update(related_list)

        while(len(self.G1.nodes()) < size):
            if(len(node_list) > 0):
                child_node = node_list.pop()
                p =  round(random.uniform(0,1),4)
                if(child_node not in dictt):
                    related_listt = list(self.g_complete.neighbors(child_node))
                    degree_c = self.g_complete.degree(child_node)
                    dictt[child_node] = child_node
                    if(p <= min(1,degree_p/degree_c) and child_node in list(self.g_complete.neighbors(parent_node))):
                        self.G1.add_edge(parent_node,child_node)
                        parent_node = child_node
                        degree_p = degree_c
                        node_list.clear()
                        node_list.update(related_listt)
                    else:
                        del dictt[child_node]


            # node_list set becomes empty or size is not reached 
            # insert some random nodes into the set for next processing
            else:
                node_list.update(random.sample(set(self.g_complete.nodes())-set(self.G1.nodes()),3))
                parent_node = node_list.pop()
                self.g_complete.add_node(parent_node)
                related_list = list(self.g_complete.neighbors(parent_node))
                node_list.clear()
                node_list.update(related_list)
        return self.G1
    
    def induced_mhrw(self,G,size,node):
        sampled_graph = self.mhrw(self.G1,G,size,node)
        induced_graph = G.subgraph(sampled_graph.nodes())
        return induced_graph
