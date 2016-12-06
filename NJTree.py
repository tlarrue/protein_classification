'''
Represents a neighbor-joining classification tree.
Equation Ref: https://en.wikipedia.org/wiki/Neighbor_joining

:attribute tree: (dict) Tree of the form {node : {neighbor : weight, ...}, ...}.
:attribute dist_matrix: (pandas DataFrame) Matrix of pairwise distances.

:function build(dist_matrix)
:function classify_treeNN(protein_sequence)
:function classify_treeInsert(protein_sequence)
'''

import decimal
import numpy as np
import pandas as pd
from pprint import pprint
import networkx as nx
import matplotlib.pyplot as plt

DEBUG = True

def _calculate_q_matrix(dist_matrix):
    # Calculates q_matrix matrix from the distance matrix (wiki EQ 1)
    n = dist_matrix.shape[0] # Number of sequences
    q_matrix = pd.DataFrame(np.nan,
                            index=dist_matrix.axes[0],
                            columns=dist_matrix.axes[1])

    # Fill in q_matrix.
    for i in range(1, q_matrix.shape[0]):
        for j in range(0, i):
            # q(i, j) = (n - 2) * dist(i, j) - sum(dist(i, )) - sum(dist( , j))
            val = (n - 2) * dist_matrix.iat[i, j]
            val -= np.nansum(dist_matrix.iloc[i, :])
            val -= np.nansum(dist_matrix.iloc[:, j])
            q_matrix.iat[i, j] = val
            q_matrix.iat[j, i] = val

    return q_matrix

def _isleaf(tree, node_name):
    if tree.node[node_name]['c'] != '':
        return True
    else:
        return False

class NJTree:

    def __init__(self):
        ''' Default constructor, initialize tree and distance matrix. '''

        self.tree = nx.Graph() #using networkx for easy visualization & analysis
        self.dist_matrix = pd.DataFrame() #using pandas for labeled matrix
        self.cluster_dictionary = {} #dict to map cluster names their group of nodes

    def cluster_leaves(self, i, j, new_cluster_name=None):
        ''' Update tree by adding a new internal node between i and j.
        
        :param i: (str) Name of first OTU being clustered.
        :param j: (str) Name of second OTU being clustered.
        :return None.
        '''
        n = self.dist_matrix.shape[0] # Number of sequences
        
        # Calculate distances from leaves to be clustered to the new node.
        # Dist from i to the new node i-j (wiki equation 2) is...
        # .5*dist(i,j) + 1/(2n-4) * (sum(dist(i, )-sum(dist(j, ))
        dist_to_i = (.5 * self.dist_matrix.at[i, j]
                    + (1.0 / (2 * n - 4))
                    * (np.nansum(self.dist_matrix.loc[i, :])
                        - np.nansum(self.dist_matrix.loc[:, j])))

        # Dist from j to new node is dist(i,j) - dist(i, i-j)
        dist_to_j = self.dist_matrix.at[i, j] - dist_to_i
        
        # Add new node to tree & attach distances to edges 
        # between each leaf and the new node
        cluster_names = list(self.cluster_dictionary.keys())
        if new_cluster_name: 
            new_node_name = new_cluster_name
        else:
            if not cluster_names:
                new_node_name = '1'
            else:
                new_node_name = str(max([int(k) for k in cluster_names]) + 1)

        [i_name, i_class] = [k.strip() for k in i.split("/")]
        [j_name, j_class] = [k.strip() for k in j.split("/")]
        self.tree.add_node(new_node_name, c='')
        self.tree.add_node(i_name, c=i_class)
        self.tree.add_node(j_name, c=j_class)
        self.tree.add_edge(i_name, new_node_name, length=dist_to_i)
        #self.tree[i][new_node_name]['distance'] = dist_to_i
        self.tree.add_edge(j_name, new_node_name, length=dist_to_j)
        #self.tree[j][new_node_name]['distance'] = dist_to_j

        # Add new node to cluster_dictionary
        self.cluster_dictionary[new_node_name] = []
        for node in [i_name,j_name]:
            if node in self.cluster_dictionary:
                self.cluster_dictionary[new_node_name].extend(self.cluster_dictionary[node]) 
            else:
                self.cluster_dictionary[new_node_name].append(node)

        return new_node_name + "/"

    def update_distances(self, i, j, node_num):
        ''' Update distance matrix by recalculating distances to/from new node.
        
        :param i: (str) Name of first OTU that was clustered.
        :param j: (str) Name of second OTU that was clustered.
        :return None.
        '''
        # Initialize new distance matrix.
        node_label = pd.Index([str(node_num)])
        new_labels = self.dist_matrix.axes[0].drop([i, j]).append(node_label)
        new_dist_matrix = pd.DataFrame(np.nan, index=new_labels, columns=new_labels)
        
        # Fill in distance matrix
        # First copy over values that stay the same
        for row in new_dist_matrix.axes[0].drop(node_label):
            for col in new_dist_matrix.axes[1].drop([node_label[0], row]):
                new_dist_matrix.at[row, col] = self.dist_matrix.at[row, col]
                new_dist_matrix.at[col, row] = self.dist_matrix.at[row, col]
                
        # Distance from other OTU, k, to new node, i-j (wiki EQ 3):
        # d(i-j, k) = .5 * (dist(i, k) + dist(j, k) - dist(i, j))
        for k in new_dist_matrix.axes[1].drop(node_label):
            dist = .5 * (self.dist_matrix.at[k, i]
                         + self.dist_matrix.at[k, j]
                         - self.dist_matrix.at[i, j])
            new_dist_matrix.at[node_label, k] = dist
            new_dist_matrix.at[k, node_label] = dist
        
        # Update the distance matrix.
        self.dist_matrix = new_dist_matrix

    def build(self, dist_matrix):
        ''' Build a classification tree via the neighbor-joining method.
        
        :param dist_matrix (pandas.DataFrame): Matrix of pairwise distances.
        :return None.
        '''
        n = dist_matrix.shape[0] # Number of sequences
        self.dist_matrix = dist_matrix

        for i in range(n - 3):
            if DEBUG:
                print 'Distance Matrix'
                pprint(self.dist_matrix)
                print

            # 1] Calculate q_matrix matrix from distances
            q_matrix = _calculate_q_matrix(self.dist_matrix)
            
            if DEBUG:
                print 'Q matrix:'
                pprint(q_matrix)
                print

            # 2] Find a pair (i,j) where q_matrix(i,j) has the lowest value
            q = q_matrix.values
            mins = np.where(q == np.nanmin(q))
            min_col_idx = mins[0][0]
            min_row_idx = mins[1][0]
            (min_col, min_row) = (q_matrix.index[min_col_idx], 
                                  q_matrix.columns[min_row_idx])

            # 3] Cluster (j, i) pair by adding new node to tree
            # min_row/min_col = sequence labels in form 'ID/class'
            new_node_name = self.cluster_leaves(min_row, min_col) 
            if DEBUG:
                print 'Tree:'
                pprint(nx.clustering(self.tree))
                pprint(self.cluster_dictionary)
                print '\n\n'

            # 4] Recalculate distances (distance matrix)
            self.update_distances(min_row, min_col, new_node_name)
            
        # Add remaining branch lengths/nodes from dist_matrix
        last_cluster_name = new_node_name.split("/")[0].strip()
        mid_edge_length = 0.5 * (self.dist_matrix.iat[0, 1]
                              + self.dist_matrix.iat[0, 2]
                              - self.dist_matrix.iat[1, 2])
        self.cluster_leaves(self.dist_matrix.columns[0], self.dist_matrix.columns[1], 'X')
        self.tree.add_edge(last_cluster_name, 'X', length=mid_edge_length)

        if DEBUG:
            print 'Final tree:'
            pprint(nx.clustering(self.tree))
            pprint(self.cluster_dictionary)
            

    def classify_treeNN(self, query_name, edge_neighborhood=3):
        '''
        Assigns label to query protein based on an analysis of 
        query's neighborhood within NJ Tree containing itself 
        and members of priori database.
        '''

        #1) Find set of closest neighbors & their class names
        # ie. leaves with at most edge_neighborhood edges between itself 
        # and the query node
        # walk thru neighbors, break when significantly above edge_neighborhood
        neighborhood = []
        neighbors = list(self.tree[query_name].keys()) # list of adjacent nodes
        
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            num_edges_between = self.tree.number_of_edges(query_name, neighbor)

            if ((num_edges_between <= edge_neighborhood) and _isleaf(self.tree,neighbor)):
                neighborhood.append(neighbor)

            if (num_edges_between > (edge_neighborhood + 2)):
                break
            else:
                # add neighbor's neighbors to 'neighbors', 
                # checking for duplicates
                for n in list(self.tree[neighbor].keys()):
                    if (n not in neighbors) and (n != query_name):
                        neighbors.append(n)
                i+=1

        
        #2) Define classes of the neighborhood {class: [ids...]}
        all_classes = [x for x in list(nx.get_node_attributes(self.tree, 'c').values()) if x not in ['query','']]
        neighborhood_classes = {c:[] for c in all_classes}
        if DEBUG: print '\nNEIGHBORHOOD: ', neighborhood
        for node in neighborhood:
            neighborhood_classes[self.tree.node[node]['c']].append(node)
        if DEBUG: print '\nNEIGHBORHOOD CLASSES: ', neighborhood_classes 

        #3) Find aggregate similarily score for each class
        # Use minimum operator for distance measure & maximum for similarity measure
        R = {}
        for c,ids in neighborhood_classes.iteritems():
            sim_score = min([nx.shortest_path_length(self.tree, source=query_name, 
                target=i, weight='length') for i in ids])
            if DEBUG: print "\tCLASS / SIM_SCORE: ", c, sim_score
            R[sim_score] = c # distance measure

        min_score = min(R.keys())
        if DEBUG: print "MIN_SCORE: ", min_score

        return R[min_score] #class of minimum distance score

    def classify_weighted_treeNN(self, query_name):
        return

    def classify_treeInsert(self, protein_sequnce):
        return


if __name__ == '__main__':
    # Create a distance matrix for testing, using the example from Wikipedia.
    labels = ['a/class1', 'b/class1', 'c/class2', 'q/query', 'e/class3'] #d is query protein - q
    dist_matrix = pd.DataFrame([[0, 5,  9,  9,  8],
                                [5, 0,  10, 10, 9],
                                [9, 10, 0,  8,  7],
                                [9, 10, 8,  0,  3],
                                [8, 9,  7,  3,  0]],
                            index=labels, columns=labels)
    
    # Build the test tree
    njt = NJTree()
    njt.build(dist_matrix)

    # Display results
    print '\nEDGES:'
    for edge in njt.tree.edges():
        print edge, ": ", njt.tree.get_edge_data(*edge)

    print '\nCLUSTER KEY:'
    pprint(njt.cluster_dictionary) 

    query_class = njt.classify_treeNN('q')
    print '\nQUERY CLASS (TreeNN): ', query_class

    labels = {i[0]: i[1]['c'] for i in njt.tree.nodes(data=True)}
    layout = nx.spring_layout(njt.tree)
    #nx.draw_networkx(njt.tree, pos=layout, with_labels=True) #ID labels
    nx.draw_networkx(njt.tree, pos=layout, with_labels=True, labels=labels) #class labels
    plt.show()



