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
        dist_to_i = (.5 * self.dist_matrix.iat[i, j] +
                    ((1 / (2 * n - 4)) *
                        (np.nansum(self.dist_matrix.iloc[i, :]) -
                        np.nansum(self.dist_matrix.iloc[:, j]))))
        # Dist from j to new node is dist(i,j) - dist(i, i-j)
        dist_to_j = self.dist_matrix.iat[i, j] - dist_to_i
        
        # Add new node to tree.
        i_name = str(self.dist_matrix.axes[0][i])
        j_name = str(self.dist_matrix.axes[0][j])
        node_name = j_name + "-" + i_name
        self.tree[node_name] = {i_name : dist_to_i, j_name : dist_to_j}
        print "Tree:"
        print self.tree


    def update_distances(self, i, j):
        # TODO: write update_distances function
        # updates the dist_matrix by replacing i & j with a new node &
        #     recalculating distances b/t new node + other OTUs & vise-versa.
        #     Also add relevant distances to the tree.
        #         wiki EQ 2 = Distance from each OTU to new node
        #         wiki EQ 3 = Distance from OTUs to new node
        return


    def build(self, dist_matrix):

        n = dist_matrix.shape[0] # Number of sequences
        self.dist_matrix = dist_matrix

        for i in range(n - 3):

            # 1] Calculate q_matrix matrix from distances
            q_matrix = _calculate_q_matrix(self.dist_matrix)

            # 2] Find a pair (i,j) where q_matrix(i,j) has the lowest value
            max = decimal.Decimal('-Infinity')
            (max_col, max_row) = (None, None)
            # TODO: Find a cleaner way to exclude last col -- need to exclude it
            # b/c it is all NaN
            nan_col = q_matrix.axes[1][q_matrix.axes[1].size - 1]
            for col in q_matrix.axes[1].drop(nan_col):
                if q_matrix[col][q_matrix.idxmax()[col]] > max:
                    max = q_matrix[col][q_matrix.idxmax()[col]]
                    (max_col, max_row) = (col, q_matrix.idxmax()[col])
            # (i,j) = q_matrix.idxmin()

            # 3] Cluster (j, i) pair by adding new node to tree
            self.cluster_leaves(int(max_row), int(max_col))

            # 4] Recalculate distances (distance matrix)
            self.update_distances(max_row, max_col)


    def classify_treeNN(self, protein_sequence):
        return


    def classify_treeInsert(self, protein_sequnce):
        return


if __name__ == '__main__':
    # Create a distance matrix for testing.
    dist_matrix = pd.DataFrame(np.nan, index=range(1, 7), columns=range(1, 7))
    val = 1
    for i in range(1, dist_matrix.shape[0]):
        for j in range(0, i):
            dist_matrix.iat[i, j] = val
            val = val + 1

    print "distance matrix:"
    print dist_matrix
    print
    print "Q matrix:"
    print _calculate_q_matrix(dist_matrix)
    print
    
    njt = NJTree()
    njt.build(dist_matrix)

