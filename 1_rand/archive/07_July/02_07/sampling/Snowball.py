import json
import sys
import random
import math 
import time
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


class Queue():
    #Constructor creates a list
    def __init__(self):
        self.queue = list()
    
    #Adding elements to queue
    def enqueue(self,data):
        #Checking to avoid duplicate entry (not mandatory)
        if data not in self.queue:
            self.queue.insert(0,data)
            return True
        return False
    
    #Removing the last element from the queue
    def dequeue(self):
        if len(self.queue)>0:
            return self.queue.pop()
        else:
            #plt.show()
            exit()
    
    #Getting the size of the queue
    def size(self):
        return len(self.queue)
    
    #printing the elements of the queue
    def printQueue(self):
        return self.queue

class Snowball():

    def __init__(self,g_complete,a_complete):
        self.G1 = nx.Graph()
        self.g_complete = g_complete
        self.a_complete = a_complete

    def snowball(self,size,k):
        q=Queue() 
        list_nodes=list(self.g_complete.nodes())
        m = k
        dictt = set()
        while(m):
            id = random.sample(list(self.g_complete.nodes()),1)[0]
            q.enqueue(id)
            m = m - 1
        #print(q.printQueue())
        start_time = time.time()
        while(len(self.G1.nodes()) <= size):
            if time.time() - start_time > 0.25:  # break if it gets stuck
                break
            if(q.size() > 0):
                id = q.dequeue()
                self.G1.add_node(id)
                if(id not in dictt):
                    dictt.add(id)
                    list_neighbors = list(self.g_complete.neighbors(id))
                    if(len(list_neighbors) > k):
                        for x in list_neighbors[:k]:
                            q.enqueue(x)
                            self.G1.add_edge(id,x)
                    elif(len(list_neighbors) <= k and len(list_neighbors) > 0):
                        for x in list_neighbors:
                            q.enqueue(x)
                            self.G1.add_edge(id,x)
                else:
                    continue
            else:
                initial_nodes = random.sample(list(self.g_complete.nodes()) and list(dictt),k)
                no_of_nodes = len(initial_nodes)
                for id in initial_nodes:
                    q.enqueue(id) 
        return self.G1


