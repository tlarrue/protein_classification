'''
Represents a neighbor-joining classification tree.
Equation Ref: https://en.wikipedia.org/wiki/Neighbor_joining
'''
import decimal
import numpy as np
import pandas as pd
import cPickle as pickle
from pprint import pprint
import networkx as nx
from sets import Set
import scipy.optimize as optimize

# Debug modes
DEBUG = False #print statements throughout steps
VIEW_ALL = False #display graphs throughout steps
PROGRESS = False #display progress in tree builds

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

def _cluster_leaves(tree, cluster_map, dist_matrix, node1, node2, class_map, new_cluster):
    """Update a tree by adding a new internal node between given nodes.
    
    Args:
        tree (networkx.Graph): Neighbor-joining tree representation.
        cluster_map (dict): map of cluster names to the nodes they enclose
        dist_matrix (pandas.DataFrame): Matrix of pairwise distances labelled with element IDs.
        node1 (str): node label of node 1.
        node2 (str): node label of node 2.
        class_map (dict): map of element IDs to their class label
        new_cluster(str): Name to assign to new internal node. 

    Returns:
        networkx.Graph: Updated Tree
        dict: Cluster Map
    """
    n = dist_matrix.shape[0] #number of sequences
    
    # Calculate distances from leaves to be clustered to the new node.
    # Dist from i to the new node i-j (wiki equation 2):
    # .5*dist(i,j) + 1/(2n-4) * (sum(dist(i, )-sum(dist(j, ))
    dist_to_node1 = (.5 * dist_matrix.at[node1, node2] + (1.0 / (2 * n - 4)) 
        * (np.nansum(dist_matrix.loc[node1, :])
            - np.nansum(dist_matrix.loc[:, node2])))

    # Dist from j to new node is dist(i,j) - dist(i, i-j)
    dist_to_node2 = dist_matrix.at[node1, node2] - dist_to_node1
    
    # Add new node to tree & attach distances to edges 
    # between each leaf and the new node
    for node in [node1,node2]:
        if node in class_map.keys():
            tree.add_node(node, c=class_map[node])
        else:
            tree.add_node(node, c='')
        
    tree.add_node(new_cluster, c='')
    tree.add_edge(node1, new_cluster, length=dist_to_node1)
    tree.add_edge(node2, new_cluster, length=dist_to_node2)

    # Add new node to cluster map
    cluster_map[new_cluster] = []
    for node in [node1,node2]:
        if node in cluster_map:
            cluster_map[new_cluster].extend(cluster_map[node]) 
        else:
            cluster_map[new_cluster].append(node)

    return tree, cluster_map

def _update_distances(dist_matrix, node1, node2, new_cluster):
    """Updates a distance matrix by recalculating distances to and from a new cluster node.

    Args:
        node1 (str): node label of node 1.
        node2 (str): node label of node 2.
        new_cluster(str): Name of cluster node of node1 and node2.
    
    Returns:
        pandas.DataFrame: Updated distance matrix.
    """
    # Initialize new distance matrix.
    node_label = pd.Index([str(new_cluster)])
    new_labels = dist_matrix.axes[0].drop([node1, node2]).append(node_label)
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
        dist = .5 * (dist_matrix.at[k, node1]
                     + dist_matrix.at[k, node2]
                     - dist_matrix.at[node1, node2])
        new_dist_matrix.at[node_label, k] = dist
        new_dist_matrix.at[k, node_label] = dist
    
    # Return the distance matrix.
    return new_dist_matrix

class NJTree:

    """Represents a neighbor-joining classification tree. 

    Attributes:
        tree (networkx.Graph): neighbor-joining tree representation.
        orig_dist_matrix (pandas.DataFrame): original distance matrix the tree is derived from.
        work_dist_matrix (pandas.DataFrame): current state of distance matrix as tree is being built
        cluster_map (dict): map of cluster names to the nodes they enclose
        class_map (dict): map of element IDs to their class label
    """

    def __init__(self):
        """Default constructor, Initialize attributes. """
        self.tree = nx.Graph() 
        self.orig_dist_matrix = pd.DataFrame()
        self.work_dist_matrix = pd.DataFrame() 
        self.cluster_map = {} 
        self.class_map = {} 

    def isLeaf(self, node_name):
        """Determines if given node is a leaf of this tree. 

        Args:
            node_name (string): node label.

        Returns:
            bool: True if node is a leaf, False otherwise.
        """
        if self.tree.node[node_name]['c'] != '':
            return True
        else:
            return False

    def cluster_leaves(self, node1, node2, new_cluster):
        """Updates this tree by adding a new internal node between given nodes.

        Updates the working distance matrix to reflect tree update by 
        recalculating distance to and from the new cluster node.

        Args:
            node1 (str): node label of node 1.
            node2 (str): node label of node 2.
            new_cluster(str): Name to assign to new internal node. 
        """
        #TODO: write function to name a new node & update _cluster_leaves
        self.tree, self.cluster_map = _cluster_leaves(self.tree, self.cluster_map, 
            self.work_dist_matrix, node1, node2, self.class_map, new_cluster)

        self.work_dist_matrix = _update_distances(self.work_dist_matrix, node1, node2, new_cluster)

    def build(self, dist_matrix, class_map, cluster_naming_function):
        """Builds this classification tree via the neighbor-joining method.

        Args:
            dist_matrix (pandas.DataFrame): Matrix of pairwise distances labelled with element IDs.
            class_map (dict): Dictionary where keys are element IDs and values are class labels.
            cluster_naming_function (function): Function to assign new names to clusters based on 
                nodes to be clustered and the cluster dictionary in form myFunct(node1, node2, cluster_map).
        """
        # Update attributes
        self.orig_dist_matrix = dist_matrix 
        self.class_map = class_map 
        self.work_dist_matrix = dist_matrix

        # Get number of elements
        n = dist_matrix.shape[0]

        if PROGRESS:
            print 'Starting tree build now!'

        # Loop through n-3 elements & add nodes in tree
        for i in range(n - 3):

            if DEBUG:
                print 'Distance Matrix'
                pprint(self.work_dist_matrix)
                print

            # Calculate q_matrix matrix from distances
            q_matrix = _calculate_q_matrix(self.work_dist_matrix)
            
            if DEBUG:
                print 'Q matrix:'
                pprint(q_matrix)
                print

            # Find pair of elements (i,j) where q_matrix(i,j) has the lowest value
            (min_col, min_row) = _find_min_pair(q_matrix)

            # Add nodes i,j, and cluster node of i and j to this tree
            # And update working distance matrix accordingly
            new_cluster_name = cluster_naming_function(min_row, min_col, self.cluster_map)
            self.cluster_leaves(min_row, min_col, new_cluster_name) 

            if DEBUG:
                print 'Tree:'
                pprint(nx.clustering(self.tree))
                pprint(self.cluster_dictionary)
                print '\n\n'
            
            # View graph after each step for debugging
            if VIEW_ALL:
                labels = {i[0]: i[0]+'/'+i[1]['c'] for i in njt.tree.nodes(data=True)}
                layout = nx.spring_layout(njt.tree)
                nx.draw_networkx(njt.tree, pos=layout, with_labels=True, labels=labels) #class labels
                plt.show()

            if PROGRESS:
                print str(i + 1) + " down, " + str(n-i-4) + " to go..."
            
        # Add remaining branch lengths and nodes from working distance matrix to this tree 
        #last_cluster_name = new_node_name.split('S')[0].strip()
        previous_cluster = new_cluster_name
        mid_edge_length = 0.5 * (self.work_dist_matrix.iat[0, 1]
                              + self.work_dist_matrix.iat[0, 2]
                              - self.work_dist_matrix.iat[1, 2])
        (node1, node2) = (self.work_dist_matrix.columns[0], self.work_dist_matrix.columns[1])
        new_cluster = cluster_naming_function(node1, node2, self.cluster_map)
        #self.cluster_leaves(self.work_dist_matrix.columns[0], self.work_dist_matrix.columns[1], 'X') 
        self.cluster_leaves(node1, node2, new_cluster)
        #self.tree.add_edge(previous_cluster, 'X', length=mid_edge_length)
        self.tree.add_edge(previous_cluster, new_cluster, length=mid_edge_length)
        #TODO: check these changes make sense, then get rid of comments

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

    def classify_treeInsert(self, query_name, cluster_naming_function):
        '''
        :param dist_matrix - a pandas Ds, excluding query 
        :param classes - a list of candidate class names for query 
        :query_name - a string representing the query ID
        '''
        classes = Set(self.class_map.values())

        full_dist_matrix = self.orig_dist_matrix.drop(query_name)
        full_dist_matrix = full_dist_matrix.drop(query_name, axis=1)

        if PROGRESS:
            print '\nStarting treeInsert!'
        
        #1] Build a tree for each class
        class_trees = {}
        all_elements = full_dist_matrix.columns.values.tolist()
        classes_done = 0
        num_of_classes = len(classes)
        for c in classes:

            #1a. Construct a mini distance matrix for the current class
            nonclass_members = [i for i in all_elements if self.class_map[i] != c]
            class_dist_matrix = full_dist_matrix.drop(nonclass_members)
            class_dist_matrix = class_dist_matrix.drop(nonclass_members, axis=1)

            #1b. Build class tree
            if PROGRESS:
                print 'Building class tree for ' + c

            class_njt = NJTree()
            class_njt.build(class_dist_matrix, self.class_map, myClusterNaming)
            class_trees[c] = class_njt
            classes_done = classes_done + 1

            if PROGRESS:
                print str(classes_done) + " classes down, " + str(num_of_classes - classes_done) + " to go..."

        #2] Determine the insertion cost of each tree
        class_insert_costs = {}
        for c,class_tree in class_trees.iteritems():

            #2a. Find insertion cost of each leaf in the tree
            leaves = [i for i in class_tree.tree.nodes() if class_tree.isLeaf(i)] #ERROR: One of nodes not splitting into
            leaf_insert_costs = {}
            for leaf_i in leaves:

                parent_i = class_tree.tree.neighbors(leaf_i)[0] #TODO: Check this is right!
                cons = ({'type': 'eq',
                         'fun': lambda x: x[0] + x[1] - nx.shortest_path_length(class_tree.tree, 
                            source=parent_i, target=leaf_i, weight='length')})
                bnds = [(0,) for x in [0,0,0]]
                optimum_leaf_insert_cost = optimize.minimize(_leaf_insertion_cost, [0,0,0], 
                    args=(class_tree.orig_dist_matrix, leaf_i, leaves, query_name, self), method='SLSQP', constraints=cons)

                if DEBUG:
                    print "Optimum cost for ", leaf_i, " : ", optimum_leaf_insert_cost.x[0]

                leaf_insert_costs[leaf_i] = optimum_leaf_insert_cost.x[0]
            
            class_insert_costs[c] = min(list(leaf_insert_costs.values()))

        #3] Output the class name of tree with minimum insertion cost
        min_insert_cost = min(list(class_insert_costs.values()))
        for c,cost in class_insert_costs.iteritems():
            if cost==min_insert_cost:
                return c
                break

def _leaf_insertion_cost(x_y_z_array, dist_matrix, leaf_i, leaves, query_name, orig_njt):
    x,y,z = x_y_z_array

    all_leaves_sum = 0.
    for leaf_j in leaves:
        if leaf_j == leaf_i:
            continue
        else:
            all_leaves_sum += orig_njt.orig_dist_matrix.at[leaf_j, query_name] - (x + z)

    return all_leaves_sum**2.

def read_distance_csv(filepath):
    f = open(filepath, 'rb')
    data = np.genfromtxt(f, delimiter=',', names=True, case_sensitive=True, dtype=None) #structured array of strings
    f.close()
    dist_matrix = pd.DataFrame(data, index=data.dtype.names,columns=data.dtype.names)
    return dist_matrix


def read_classes_csv(filepath):
    f = open(filepath, 'rb')
    data = np.genfromtxt(f, delimiter=',', names=True, case_sensitive=True, dtype=None)
    f.close()
    df = pd.DataFrame(data)
    # Replace ' ' with '_' to match IDs from distance matrix (IDs are forced to
    # have '_' in dist_matrix because they are the labels for it)
    df['ID'] = df['ID'].str.replace(' ','_')
    return df.set_index('ID')['Class'].to_dict()

def myClusterNaming(node1, node2, cluster_map):
    cluster_names = list(cluster_map.keys())
    if not cluster_names:
        new_node_name = '1'
    else:
        new_node_name = str(max([int(k) for k in cluster_names]) + 1)

    return new_node_name

def perform_test(test_set):
    '''
    test_set Form: {'tree': name_of_example,
                    'viz': true or false,
                    'classification': [TreeNN, TreeInsert]}
    '''
    #Define a test tree    
    tree_built = False
    if test_set['tree'] == "wiki":

        print "\nBuilding Tree with Wikipedia example...\n"
        labels = ['a', 'b', 'c', 'q', 'e'] #d is query protein - q
        class_map = {'a':'class1','b':'class1','c':'class2','q':'query','e':'class3'}
        dist_matrix = pd.DataFrame([[0, 5,  9,  9,  8],
                                   [5, 0,  10, 10, 9],
                                   [9, 10, 0,  8,  7],
                                   [9, 10, 8,  0,  3],
                                   [8, 9,  7,  3,  0]],
                               index=labels, columns=labels)
        query_name = 'q'

    elif test_set['tree'] == "protein_database":

        print "\nBuilding Tree with data/distance_matrix.csv & data/id_lookup.csv...\n"
        dist_matrix = read_distance_csv('./data/distance_matrix.csv')
        class_map = read_classes_csv('./data/id_lookup.csv')
        query_name = class_map.keys()[0]

    else:

        print "\nUsing built tree from file: " + test_set['tree'] + " ...\n"
        njt = pickle.load(open(test_set['tree'], "rb" ))
        query_name = njt.class_map.keys()[0]
        tree_built=True

    #Build the test tree
    if not tree_built:
        njt = NJTree()
        njt.build(dist_matrix, class_map, myClusterNaming)

    #visualize tree
    #TODO: work on best viz, color nodes by class name
    if test_set['viz']:

        labels = {i[0]: i[1]['c'] for i in njt.tree.nodes(data=True)}
        layout = nx.spring_layout(njt.tree)
        #nx.draw_networkx(njt.tree, pos=layout, with_labels=True) #ID labels
        nx.draw_networkx(njt.tree, with_labels=True, labels=labels, node_size=100) #class labels
        plt.show()

    query_results = {}

    classifications = [i.lower() for i in test_set['classification']]
    if "treenn" in classifications:

        query_class = njt.classify_treeNN(query_name)
        query_results['TreeNN'] = query_class
        print '\nQUERY CLASS (TreeNN): ', query_class

        query_class = njt.classify_weighted_treeNN(query_name)
        query_results['Weighted_TreeNN'] = query_class
        print '\nQUERY CLASS (Weighted TreeNN): ', query_class

    if "treeinsert" in classifications:
        query_class = njt.classify_treeInsert(query_name, myClusterNaming)
        query_results['TreeInsert'] = query_class
        print '\nQUERY CLASS (TreeInsert): ', query_class

    return NJTree, query_results

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    myTest = {'tree': "./data/protein_db_njtree.p",
    #myTest = {'tree': "protein_database",
              'viz': False,
              'classification': ['TreeNN', 'TreeInsert']}

    myTree, results = perform_test(myTest)

    #save_file = open("./data/protein_db_njtree.p", "wb")
    #pickle.dump(myTree, save_file)
    #save_file.close()
