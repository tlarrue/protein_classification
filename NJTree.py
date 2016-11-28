'''
Represents a neighbor-joining classification tree. 
Equation Ref: https://en.wikipedia.org/wiki/Neighbor_joining

Attributes:
- tree [nested dictionaries?]

Methods:
- build(distance_matrix)
- classify_treeNN(protein_sequence)
- classify_treeInsert(protein_sequnce)
'''

import decimal
import numpy as np
import pandas as pd

def _calculate_q_matrix(distance_matrix):
    # Calculates q_matrix matrix from the distance matrix
    # wiki EQ 1
    n = distance_matrix.shape[0] # Number of sequences
    q_matrix = pd.DataFrame(np.nan,
                            index=distance_matrix.axes[0],
                            columns=distance_matrix.axes[1])

    # fill in q_matrix
    for i in range(1, q_matrix.shape[0]):
        for j in range(0, i):
            # q(i, j) = (n - 2) * dist(i, j) - sum(dist(i, )) - sum(dist( ,j))
            val = (n - 2) * distance_matrix.iat[i, j]
            val -= np.nansum(distance_matrix.iloc[:, j])
            val -= np.nansum(distance_matrix.iloc[i, :])
            q_matrix.iat[i, j] = val

    return q_matrix


class NJTree:

    def __init__(self):
        # TODO: decide on data structure
        self.distance_matrix = [] # Pandas DS? -- DataFrame?
        # nested dictionaries? - must include lengths
        # of the form {node1# : {neighbor1# : weight, ...}, ...} ?
        self.tree = {}


    def cluster_leaves(self, i, j):
        # TODO: write cluster_leaves funtion
        # inputs: 2 leaves to be clustered
        # updates tree by adding a new internal node to the tree between i & j
        return


    def update_distances(self, i, j):
        # TODO: write update_distances function
        # updates the distance_matrix by replacing i & j with a new node &
        #     recalculating distances b/t new node + other OTUs & vise-versa.
        #     Also add relevant distances to the tree.
        #         wiki EQ 2 = Distance from each OTU to new node
        #         wiki EQ 3 = Distance from OTUs to new node
        return


    def build(self, distance_matrix):

        n = distance_matrix.shape[0] # Number of sequences
        self.distance_matrix = distance_matrix

        for i in range(n - 3):

            # 1] Calculate q_matrix matrix from distances
            q_matrix = _calculate_q_matrix(self.distance_matrix)

            # 2] Find a pair (i,j) where q_matrix(i,j) has the lowest value
            max = decimal.Decimal('-Infinity')
            (max_col, max_row) = (None, None)
            # TODO: Find a cleaner way to exclude last col -- need to exclude it
            # b/c it is all NaN
            for col in q_matrix.axes[1].drop(q_matrix.axes[1][q_matrix.axes[1].size - 1]):
                if q_matrix[col][q_matrix.idxmax()[col]] > max:
                    max = q_matrix[col][q_matrix.idxmax()[col]]
                    (max_col, max_row) = (col, q_matrix.idxmax()[col])
            # (i,j) = q_matrix.idxmin()

            # 3] Cluster (j, i) pair by adding new node to tree
            self.cluster_leaves(max_row, max_col)

            # 4] Recalculate distances (distance matrix)
            self.update_distances(max_row, max_col)


    def classify_treeNN(self, protein_sequence):
        return


    def classify_treeInsert(self, protein_sequnce):
        return


if __name__ == '__main__':
    # create a distance matrix for testing
    distance_matrix = pd.DataFrame(np.nan, index=range(1, 7), columns=range(1, 7))
    val = 1
    for i in range(1, distance_matrix.shape[0]):
        for j in range(0, i):
            distance_matrix.iat[i, j] = val
            val = val + 1

    print "distance matrix:"
    print distance_matrix
    print
    print "Q matrix:"
    print _calculate_q_matrix(distance_matrix)
    
    njt = NJTree()
    njt.build(distance_matrix)

