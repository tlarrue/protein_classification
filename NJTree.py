'''
Represents a neighbor-joining classification tree. 
Equation Ref: https://en.wikipedia.org/wiki/Neighbor_joining

Attributes:
- tree [nested dictionaries?]

Methods:
- build(dist_matrix)
- classify_treeNN(protein_sequence)
- classify_treeInsert(protein_sequnce)
'''

import decimal
import numpy as np
import pandas as pd
from pprint import pprint

def _calculate_q_matrix(dist_matrix):
    # Calculates q_matrix matrix from the distance matrix
    # wiki EQ 1
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
        # TODO: decide on data structure
        self.dist_matrix = [] # Pandas DS? -- DataFrame?
        # nested dictionaries? - must include lengths
        # of the form {node1# : {neighbor1# : weight, ...}, ...} ?
        self.tree = {}


    def cluster_leaves(self, i, j):
        # inputs: 2 leaves to be clustered
        # Updates tree by adding a new internal node to the tree between i & j.
        
        n = dist_matrix.shape[0] # Number of sequences
        
        # Calculate distances from leaves to be clustered to the new node.
        # Dist from i to the new node i-j is...
        # .5*dist(i,j) + 1/(2n-4) * (sum(dist(i, )-sum(dist(j, ))
        dist_to_i = (.5 * self.dist_matrix.at[i, j] +
                    ((1 / (2 * n - 4)) *
                        (np.nansum(self.dist_matrix.loc[i, :]) -
                        np.nansum(self.dist_matrix.loc[:, j]))))
        # Dist from j to new node is dist(i,j) - dist(i, i-j)
        dist_to_j = self.dist_matrix.at[i, j] - dist_to_i
        
        # Add new node to tree.
        node_name = "(" + j + "-" + i + ")"
        self.tree[node_name] = {i : dist_to_i, j : dist_to_j}
        if self.tree.has_key(i):
            self.tree[i][node_name] = dist_to_i
        if self.tree.has_key(j):
            self.tree[j][node_name] = dist_to_j
        print "Tree:"
        pprint(self.tree)
        print


    def update_distances(self, i, j):
        # updates the dist_matrix by replacing i & j with a new node &
        #     recalculating distances b/t new node + other OTUs & vise-versa.
        #     Also add relevant distances to the tree.
        #         wiki EQ 2 = Distance from each OTU to new node
        #         wiki EQ 3 = Distance from OTUs to new node
        
        node_label = pd.Index(["("  + j + "-" + i + ")"])
        new_labels = self.dist_matrix.axes[0].drop([i, j]).append(node_label)
        new_dist_matrix = pd.DataFrame(np.nan, index=new_labels, columns=new_labels)
        
        # Fill in distance matrix
        # First copy over values that stay the same
        for row in new_dist_matrix.axes[0].drop(node_label):
            for col in new_dist_matrix.axes[1].drop([node_label[0], row]):
                new_dist_matrix.at[row, col] = self.dist_matrix.at[row, col]
                new_dist_matrix.at[col, row] = self.dist_matrix.at[row, col]
                
        # Distance from other OTU, k, to new node, i-j:
        # d(i-j, k) = .5 * (dist(i, k) + dist(j, k) - dist(i, j))
        for k in new_dist_matrix.axes[1].drop(node_label):
            dist = .5 * (self.dist_matrix.at[k, i]
                         + self.dist_matrix.at[k, j]
                         - self.dist_matrix.at[i, j])
            new_dist_matrix.at[node_label, k] = dist
            new_dist_matrix.at[k, node_label] = dist
        print "New distance matrix:"
        pprint(new_dist_matrix)
        print
        self.dist_matrix = new_dist_matrix


    def build(self, dist_matrix):

        n = dist_matrix.shape[0] # Number of sequences
        self.dist_matrix = dist_matrix

        for i in range(n - 3):

            # 1] Calculate q_matrix matrix from distances
            q_matrix = _calculate_q_matrix(self.dist_matrix)
            print "Q matrix:"
            pprint(q_matrix)
            print

            # 2] Find a pair (i,j) where q_matrix(i,j) has the lowest value
            min = decimal.Decimal('Infinity')
            (min_col, min_row) = (None, None)
            
            # TODO: Find a cleaner way to exclude last col -- need to exclude it
            # b/c it is all NaN
            nan_col = q_matrix.axes[1][q_matrix.axes[1].size - 1]
            for col in q_matrix.axes[1].drop(nan_col):
                if q_matrix[col][q_matrix.idxmin()[col]] < min:
                    min = q_matrix[col][q_matrix.idxmin()[col]]
                    (min_col, min_row) = (col, q_matrix.idxmin()[col])
            # (i,j) = q_matrix.idxmin()

            # 3] Cluster (j, i) pair by adding new node to tree
            self.cluster_leaves(min_row, min_col)

            # 4] Recalculate distances (distance matrix)
            self.update_distances(min_row, min_col)
            
        # Add remaining branch lengths/nodes from dist_matrix
        first_branch = 0.5 * (self.dist_matrix.iat[0, 1]
                              + self.dist_matrix.iat[0, 2]
                              - self.dist_matrix.iat[1, 2])
        second_branch = self.dist_matrix.iat[0, 1] - first_branch
        third_branch = self.dist_matrix.iat[1, 2] - second_branch
        
        self.tree["X"] = {self.dist_matrix.axes[0][0] : first_branch,
                        self.dist_matrix.axes[0][1] : second_branch,
                        self.dist_matrix.axes[0][2] : third_branch}
        for otu in self.dist_matrix.axes[0]:
            if self.tree.has_key(otu):
                self.tree[otu]["X"] = self.tree["X"][otu]
        
        print "Tree:"
        pprint(self.tree)
        print
            


    def classify_treeNN(self, protein_sequence):
        return


    def classify_treeInsert(self, protein_sequnce):
        return


if __name__ == '__main__':
    # Create a distance matrix for testing.
    labels = ["1", "2", "3", "4", "5", "6"]
    dist_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    val = 1
    for i in range(1, dist_matrix.shape[0]):
        for j in range(0, i):
            dist_matrix.iat[i, j] = val
            dist_matrix.iat[j, i] = val
            val = val + 1

    print "distance matrix:"
    pprint(dist_matrix)
    print
    
    njt = NJTree()
    njt.build(dist_matrix)

