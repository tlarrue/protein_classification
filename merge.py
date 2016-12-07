import numpy as np

if __name__ == '__main__':

	f = open('./id_lookup.csv', 'rb')
	lookup_data = np.genfromtxt(f, delimiter=',', names=True, case_sensitive=False, dtype=None) 

	f = open('./distance_matrix.csv', 'rb')
	matrix = np.genfromtxt(f, delimiter=',', names=True, case_sensitive=False, dtype=None)
	f.close()

	print lookup_data.dtype.names
	for ind,label in enumerate(matrix.dtype.names):
		c = lookup_data['Class'][lookup_data['ID'] == label]
		matrix.dtype.names[ind] = c

	np.savetxt('./distance_matrix_with_classes.csv', matrix, delimiter=",", 
		header=",".join(i for i in array.dtype.names), comments="", fmt='%s')
