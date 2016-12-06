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


class NJTree:

    def __init__(self):
        ''' Default constructor, initialize tree and distance matrix. '''
        # Tree is made up of nested dictionaries of the form
        #     {node_1_name : {neighbor_1_name : weight, ...}, ...}
        self.tree = {}
        # Distance matrix is a pandas DataFrame b/c it is a labeled
        self.dist_matrix = pd.DataFrame()


    def cluster_leaves(self, i, j):
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
        
        # Add new node to tree.
        node_name = '(' + j + '-' + i + ')'
        self.tree[node_name] = {i : dist_to_i, j : dist_to_j}
        if self.tree.has_key(i):
            self.tree[i][node_name] = dist_to_i
        if self.tree.has_key(j):
            self.tree[j][node_name] = dist_to_j


    def update_distances(self, i, j):
        ''' Update distance matrix by recalculating distances to/from new node.
        
        :param i: (str) Name of first OTU that was clustered.
        :param j: (str) Name of second OTU that was clustered.
        :return None.
        '''
        # Initialize new distance matrix.
        node_label = pd.Index(['('  + j + '-' + i + ')'])
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
            self.cluster_leaves(min_row, min_col)
            if DEBUG:
                print 'Tree:'
                pprint(self.tree)
                print '\n\n'

            # 4] Recalculate distances (distance matrix)
            self.update_distances(min_row, min_col)
            
        # Add remaining branch lengths/nodes from dist_matrix
        first_branch = 0.5 * (self.dist_matrix.iat[0, 1]
                              + self.dist_matrix.iat[0, 2]
                              - self.dist_matrix.iat[1, 2])
        second_branch = self.dist_matrix.iat[0, 1] - first_branch
        third_branch = self.dist_matrix.iat[1, 2] - second_branch
        
        self.tree['X'] = {self.dist_matrix.axes[0][0] : first_branch,
                        self.dist_matrix.axes[0][1] : second_branch,
                        self.dist_matrix.axes[0][2] : third_branch}
                        
        for otu in self.dist_matrix.axes[0]:
            if self.tree.has_key(otu):
                self.tree[otu]['X'] = self.tree['X'][otu]
        
        if DEBUG:
            print 'Final tree:'
            pprint(self.tree)
            

    def classify_treeNN(self, protein_sequence):
        return


    def classify_treeInsert(self, protein_sequnce):
        return


if __name__ == '__main__':
    # Create a distance matrix for testing, using the example from Wikipedia.
    labels = ['a', 'b', 'c', 'd', 'e']
    dist_matrix = pd.DataFrame([[0, 5,  9,  9,  8],
                                [5, 0,  10, 10, 9],
                                [9, 10, 0,  8,  7],
                                [9, 10, 8,  0,  3],
                                [8, 9,  7,  3,  0]],
                            index=labels, columns=labels)
    
    # Build the test tree
    njt = NJTree()
    njt.build(dist_matrix)

