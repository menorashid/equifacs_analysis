import os
import numpy as np
from helpers import util, visualize
import networkx as nx

def bfs(graph, start_node):
    que = []
    tree = []
    que.append(start_node)
    while True:
        curr_node = que[0]
        print que
        print tree
        print curr_node
        print graph[curr_node,:]
        neighbors = list(np.where(graph[curr_node,:]>0)[0])
        print neighbors

        for neighbor in neighbors:
            if (neighbor not in tree) and (neighbor not in que):
                que.append(neighbor)
        tree.append(curr_node)
        if len(que)==1:
            break
        que = que[1:]
        raw_input()
    return tree

def get_max_clique(graph):
    all_cliques = []
    sizes = []
    weights = []
    arr_start_nodes = list(range(graph.shape[0]))
    for start_node in arr_start_nodes:
        new_clique = bfs(graph, start_node)
        print graph
        print start_node
        print new_clique
        raw_input()
        all_cliques.append(new_clique)

        sizes.append(len(new_clique))
        
        # new_clique = np.array(new_clique)
        # weight_sum = graph[new_clique,:]
        # weight_sum = graph[:,new_clique]
        # weight_sum = np.sum(weight_sum)/2.
        # weights.append(weight_sum)


def main():
    cooc_diff_org = np.load( '../experiments/cooc_simple/cooc_mat_30.npy')
    classes = np.load( '../experiments/cooc_simple/classes_30.npy')
    print 'done loading'
    # print cooc_diff_org.shape
    # print classes.shape
    print 'hello'

    cooc_bin = (cooc_diff_org>0).astype(int)
    cooc_up = np.triu(cooc_bin)
    cooc_down = np.tril(cooc_bin)
    # print cooc_bin[:5,:5]
    # print cooc_up[:5,:5]
    # print cooc_down[:5,:5]
    # print cooc_down.T[:5,:5]
    # cooc_bin = cooc_up*cooc_down.T
    cooc_bin = (cooc_up+cooc_down.T)>1

    # print cooc_bin[:5,:5]
    G = nx.from_numpy_matrix(cooc_bin)
    # , create_using=nx.DiGraph)
    # print list(G.edges)
    # raw_input()
    # G = G.to_undirected(G)
    cliques =  list(nx.find_cliques(G))
    clique_idx = []
    clique_sum = []
    for idx_k, k in enumerate(cliques):
        
        k.sort()
        # print k
        
        edges = cooc_diff_org[k,:]
        edges = edges[:,k]
        edges[edges<0]=0
        # if idx_k==0:
        print edges
            # x = cooc_bin[k,:]

        # edges = np.triu(edges)
        # print edges
        print idx_k, k, np.sum(edges)
        clique_idx.append(k)
        clique_sum.append(np.sum(edges))
        


    max_idx = np.argmax(clique_sum)
    max_clique = clique_idx[max_idx]
    print classes[max_clique]
    # [4, 11, 9, 7, 17, 13, 14, 20, 5]]
    # get_max_clique(cooc_diff_org)

if __name__=='__main__':
    main()