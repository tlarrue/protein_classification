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

def _calculate_Qmatrix(distance_matrix):
	# TODO: write Q matrix function
	# Calculates Q matrix from the distance matrix
	# wiki EQ 1
	return Q

class NJtree:

	def __init__(self):
		#TODO: decide on data structure
		self.distance_matrix = [] # Pandas DS?
		self.tree = {} #nested dictionaries? - must include lengths

	def cluster_leaves(self, i, j):
		# TODO: write cluster_leaves funtion
		# inputs: 2 leaves to be clustered
		# updates tree by adding a new internal node to the tree between i & j

	def update_distances(self, i, j):
		# TODO: write update_distances function
		# updates the distance_matrix by replacing i & j with a new node & 
		# 	recalculating distances b/t new node + other OTUs & vise-versa.
		#	Also add relevant distances to the tree.
		# 		wiki EQ 2 = Distance from each OTU to new node
		# 		wiki EQ 3 = Distance from OTUs to new node 


	def build(self, distance_matrix): # consider pandas DS for labeled 2-D matrix

		N = distances.shape[0] #number of sequences

		for i in range(N-3):

			# 1] Calculate Q matrix from distances
			Q = _calculate_Qmatrix(self.distance_matrix)
 
			# 2] Search Q to find a pair (i,j) where Q(i,j) has the lowest value
			(i,j) = Q.idxmin() 

			# 3] Cluster (i,j) pair by adding new node to tree 
			self.cluster_leaves(i,j)

			# 4] Recalculate distances (distance matrix)
			self.update_distances(i,j)
				
	def classify_treeNN(self, protein_sequence):

	def classify_treeInsert(self, protein_sequnce):

