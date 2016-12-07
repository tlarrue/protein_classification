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

DEBUG = False

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

def _find_min_pair(pandas_matrix):
    '''Returns column/row header pair of the lowest cell in a pandas matrix'''
    numpy_matrix = pandas_matrix.values
    mins = np.where(numpy_matrix == np.nanmin(numpy_matrix))
    min_col_idx = mins[0][0]
    min_row_idx = mins[1][0]
    (min_col, min_row) = (pandas_matrix.index[min_col_idx], 
                          pandas_matrix.columns[min_row_idx])

    return (min_col, min_row)

def _cluster_leaves(tree, cluster_dict, dist_matrix, i, j, new_cluster_name=None):
    ''' Update tree by adding a new internal node between i and j.
    
    :param tree
    :param dist_matrix: current state of distance matrix
    :param i: (str) Name of first OTU being clustered.
    :param j: (str) Name of second OTU being clustered.
    :return tree, cluster_dict, dist_matrix, new_node_name.
    '''
    n = dist_matrix.shape[0] # Number of sequences
    
    # Calculate distances from leaves to be clustered to the new node.
    # Dist from i to the new node i-j (wiki equation 2) is...
    # .5*dist(i,j) + 1/(2n-4) * (sum(dist(i, )-sum(dist(j, ))
    dist_to_i = (.5 * dist_matrix.at[i, j]
                + (1.0 / (2 * n - 4))
                * (np.nansum(dist_matrix.loc[i, :])
                    - np.nansum(dist_matrix.loc[:, j])))

    # Dist from j to new node is dist(i,j) - dist(i, i-j)
    dist_to_j = dist_matrix.at[i, j] - dist_to_i
    
    # Add new node to tree & attach distances to edges 
    # between each leaf and the new node
    cluster_names = list(cluster_dict.keys())
    if new_cluster_name: 
        new_node_name = new_cluster_name
    else:
        if not cluster_names:
            new_node_name = '1'
        else:
            new_node_name = str(max([int(k) for k in cluster_names]) + 1)

    [i_name, i_class] = [k.strip() for k in i.split("/")]
    [j_name, j_class] = [k.strip() for k in j.split("/")]
    tree.add_node(new_node_name, c='')
    tree.add_node(i_name, c=i_class)
    tree.add_node(j_name, c=j_class)
    tree.add_edge(i_name, new_node_name, length=dist_to_i)
    tree.add_edge(j_name, new_node_name, length=dist_to_j)

    # Add new node to cluster_dictionary
    cluster_dict[new_node_name] = []
    for node in [i_name,j_name]:
        if node in cluster_dict:
            cluster_dict[new_node_name].extend(cluster_dict[node]) 
        else:
            cluster_dict[new_node_name].append(node)

    return tree, cluster_dict, dist_matrix, new_node_name + "/"

def _update_distances(dist_matrix, i, j, new_node_name):
    ''' Update distance matrix by recalculating distances to/from new node.
    
    :param i: (str) Name of first OTU that was clustered.
    :param j: (str) Name of second OTU that was clustered.
    :return None.
    '''
    # Initialize new distance matrix.
    node_label = pd.Index([str(new_node_name)])
    new_labels = dist_matrix.axes[0].drop([i, j]).append(node_label)
    new_dist_matrix = pd.DataFrame(np.nan, index=new_labels, columns=new_labels)
    
    # Fill in distance matrix
    # First copy over values that stay the same
    for row in new_dist_matrix.axes[0].drop(node_label):
        for col in new_dist_matrix.axes[1].drop([node_label[0], row]):
            new_dist_matrix.at[row, col] = dist_matrix.at[row, col]
            new_dist_matrix.at[col, row] = dist_matrix.at[row, col]
            
    # Distance from other OTU, k, to new node, i-j (wiki EQ 3):
    # d(i-j, k) = .5 * (dist(i, k) + dist(j, k) - dist(i, j))
    for k in new_dist_matrix.axes[1].drop(node_label):
        dist = .5 * (dist_matrix.at[k, i]
                     + dist_matrix.at[k, j]
                     - dist_matrix.at[i, j])
        new_dist_matrix.at[node_label, k] = dist
        new_dist_matrix.at[k, node_label] = dist
    
    # Return the distance matrix.
    return new_dist_matrix

def _isLeaf(tree, node_name):
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

    def isLeaf(self, node_name):
        ''' Determines if given node is a leaf of this tree. '''
        return _isLeaf(self.tree, node_name)

    def cluster_leaves(self, i, j, new_cluster_name=None):
        ''' Update this tree by adding a new internal node between i and j '''

        self.tree, self.cluster_dictionary, self.dist_matrix, new_node_name = _cluster_leaves(
            self.tree, self.cluster_dictionary, self.dist_matrix, i, j, new_cluster_name)

        return new_node_name

    def update_distances(self, i, j, new_node_name):
        ''' Update this distance matrix by recalculating distances to/from new node'''

        self.dist_matrix = _update_distances(self.dist_matrix, i, j, new_node_name)

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
            (min_col, min_row) = _find_min_pair(q_matrix)

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

    def getNeighborhoodClasses(self, node_name, max_edges=3):
        ''' Returns dictionary of classes in neighborhood surrounding given node.
        Dictionary is keyed by class name and values are a list of node names 
        within each class '''

        neighborhood = [] #build a list of nodes in the neighborhood of input node
        
        # Walk through adjacent nodes 1 level at a time, 
        # testing if they are in the neighborhood
        neighbors = list(self.tree[node_name].keys()) #list of nodes adjacent to input node
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            num_edges_between = self.tree.number_of_edges(node_name, neighbor)

            #neighborhood conditions:
            if ((num_edges_between <= max_edges) and self.isLeaf(neighbor)):
                neighborhood.append(neighbor)
 
            if (num_edges_between > (max_edges + 2)):
                #break if beyond neighborhood
                break
            else:
                #add next level of adjacent nodes to list of nodes to test
                neighbors.extend([n for n in list(self.tree[neighbor].keys()) if n not in neighbors+[node_name]])
                i+=1

        if DEBUG: print '\nNEIGHBORHOOD: ', neighborhood      

        # Get classes of the neighborhood in form: {class: [nodes...]}
        neighborhood_classes = {}
        for node in neighborhood:
            node_class = self.tree.node[node]['c']
            if node_class not in neighborhood_classes:
                neighborhood_classes[node_class] = []
            neighborhood_classes[node_class].append(node)

        if DEBUG: print '\nNEIGHBORHOOD CLASSES: ', neighborhood_classes 

        return neighborhood_classes

    def classify_treeNN(self, query_name, neighborhood_max_edges=3):
        '''
        Assigns label to query protein based on an analysis of 
        query's neighborhood within NJ Tree containing itself 
        and members of priori database.
        '''

        # 1) Find set of closest neighbors & their class names
        # ie. leaves with at most neighborhood_max_edges edges between itself 
        # and the query node
        neighborhood_classes = self.getNeighborhoodClasses(query_name, neighborhood_max_edges)

        # 2) Find aggregate similarity score for each class
        # Use minimum operator for distance measure & maximum for similarity measure
        # EQ 6.1 in Chapt 6, Busa-Fekete et al
        R = {}
        for c,ids in neighborhood_classes.iteritems():
            sim_score = min([nx.shortest_path_length(self.tree, source=query_name, 
                target=i, weight='length') for i in ids])
            if DEBUG: print "\tCLASS / SIM_SCORE: ", c, sim_score
            R[sim_score] = c # distance measure

        min_score = min(R.keys())
        if DEBUG: print "MIN_SCORE: ", min_score

        return R[min_score] #class of minimum distance score

    def classify_weighted_treeNN(self, query_name, neighborhood_max_edges=3):
        '''Varient of classify_treeNN, that divides the similarity 
        scores by path length to increase influence of tree structure 
        on class assignment of query sequence'''

        # 1) Find set of closest neighbors & their class names
        # ie. leaves with at most neighborhood_max_edges edges between itself 
        # and the query node
        neighborhood_classes = self.getNeighborhoodClasses(query_name, neighborhood_max_edges)

        # 2) Find aggregate weighted similarity score for each class
        # Use minimum operator for distance measure & maximum for similarity measure
        # EQ 6.3 in Chapt 6, Busa-Fekete et al
        R = {}
        for c,ids in neighborhood_classes.iteritems():
            sim_score = min([(nx.shortest_path_length(self.tree, source=query_name, 
                target=i, weight='length')/nx.shortest_path_length(self.tree, source=query_name, 
                target=i)) for i in ids])
            if DEBUG: print "\tCLASS / SIM_SCORE: ", c, sim_score
            R[sim_score] = c # distance measure

        min_score = min(R.keys())
        if DEBUG: print "MIN_SCORE: ", min_score

        return R[min_score] #class of minimum distance score

    def classify_treeInsert(self, full_dist_matrix, classes, query):
        '''
        :param dist_matrix - a pandas Ds, excluding query 
        :param classes - a list of candidate class names for query 
        :query_name - a string representing the query ID
        '''

        orig_dist_matrix = full_dist_matrix 

        full_dist_matrix = full_dist_matrix.drop(query)
        full_dist_matrix = full_dist_matrix.drop(query, axis=1)
        query_name = query.split("/")[0].strip()

        #1] Build a tree for each class
        class_trees = {}
        all_columns = full_dist_matrix.columns.values.tolist()

        for c in classes:

            #1a. Construct a mini distance matrix for the current class
            nonclass_members = [i for i in all_columns if (c not in i)]
            class_dist_matrix = full_dist_matrix.drop(nonclass_members)
            class_dist_matrix = class_dist_matrix.drop(nonclass_members, axis=1)

            #1b] Loop through n-3 distance matrix elements & add nodes in tree
            class_tree = nx.Graph()
            class_cluster_dict = {}
            n = class_dist_matrix.shape[0] #number of members in class
            
            for i in range(n - 3):

                # i] Calculate q_matrix matrix from distances
                class_q_matrix = _calculate_q_matrix(class_dist_matrix)

                # ii] Find a pair (i,j) where q_matrix(i,j) has the lowest value
                (min_col, min_row) = _find_min_pair(class_q_matrix)

                # iii] Cluster (j, i) pair by adding new node to tree
                # min_row & min_col = sequence labels in form 'ID/class'
                class_tree, class_cluster_dict, class_dist_matrix, new_node_name = _cluster_leaves(
                    class_tree, class_cluster_dict, class_dist_matrix, min_row, min_col)

                # iv] Recalculate distances in distance matrix
                class_dist_matrix = _update_distances(class_dist_matrix, min_row, 
                    min_col, new_node_name)
                

            # 1c] Add remaining branch lengths/nodes from distance matrix
            if (n > 3):
                last_cluster_name = new_node_name.split("/")[0].strip()
                mid_edge_length = 0.5 * (class_dist_matrix.iat[0, 1]
                                      + class_dist_matrix.iat[0, 2]
                                      - class_dist_matrix.iat[1, 2])
            else:
                last_cluster_name = class_dist_matrix.columns.values.tolist()[0]
                mid_edge_length = 0.5 * class_dist_matrix.iat[0, 1]
                
            class_tree, class_cluster_dict, class_dist_matrix, new_node_name = _cluster_leaves(
                class_tree, class_cluster_dict, class_dist_matrix, class_dist_matrix.columns[0], 
                class_dist_matrix.columns[1], 'X')
            class_tree.add_edge(last_cluster_name, 'X', length=mid_edge_length)

            #1d] Add class tree to dictionary
            print c
            print class_tree.nodes(data=True)
            class_trees[c] = class_tree

        #2] Determine the insertion cost of each tree
        class_insert_costs = pd.DataFrame(np.zeros(len(classes)), index=classes)

        for c,class_tree in class_trees.iteritems():

            #2a. Find insertion cost of each leaf in the tree
            print nx.clustering(class_tree)
            print class_tree.nodes(data=True)
            toto=raw_input()
            leaves = [i for i in class_tree.nodes() if _isLeaf(class_tree,i)] #ERROR: One of nodes not splitting into
            leaf_insert_costs = pd.DataFrame(np.zeros(len(leaves)), index=leaves)

            for leaf_i in leaves:

                optimum_insertion_cost = 100

                other_leaves = 0.
                for leaf_j in leaves:
                    other_leaves += orig_dist_matrix[leaf_j, query_name] - nx.shortest_path_length(
                        class_tree, source=leaf_i, target=query_name, weight='length')

                leaf_insert_costs[leaf_i] = min(other_leaves**2)
            
            class_insert_costs[c] = min(leaf_insert_costs)

        #3] Output the class name of tree with minimum insertion cost
        idx = class_insert_costs.idxmin(axis=1)

        return idx

def readDistanceCSV(filepath):
    f = open(filepath, 'rb')
    data = np.genfromtxt(f, delimiter=',', names=True, case_sensitive=False, dtype=None) #structured array of strings
    f.close()
    dist_matrix = pd.DataFrame(data, index=data.dtype.names,columns=labels)
    return dist_matrix

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

    query_class = njt.classify_weighted_treeNN('q')
    print 'QUERY CLASS (Weighted TreeNN): ', query_class

    #classes = ['class1', 'class2', 'class3']
    #query_class = njt.classify_treeInsert(dist_matrix, classes, 'q/query')
    #print 'QUERY CLASS (TreeInsert): ', query_class

    labels = {i[0]: i[1]['c'] for i in njt.tree.nodes(data=True)}
    layout = nx.spring_layout(njt.tree)
    #nx.draw_networkx(njt.tree, pos=layout, with_labels=True) #ID labels
    nx.draw_networkx(njt.tree, pos=layout, with_labels=True, labels=labels) #class labels
    plt.show()



