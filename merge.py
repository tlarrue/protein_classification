import numpy as np

if __name__ == '__main__':

	f = open('./id_lookup.csv', 'rb')
	lookup_data = np.genfromtxt(f, delimiter=",", names=True, case_sensitive=False, dtype=None) 

	f = open('./distance_matrix.csv', 'rb')
	matrix = np.genfromtxt(f, delimiter=",", names=True, case_sensitive=False, dtype=None)
	f.close()

	new_labels = []
	for label in list(matrix.dtype.names):

		try:
			c = label + "-" + lookup_data['CLASS'][np.where(lookup_data['ID'] == label.replace("_", " "))][0]
		except IndexError:
			new_labels.append(label)
			print "DELETE", label
		else:
			new_labels.append(c)

	new_matrix = np.array(matrix, dtype=[(i,'f8') for i in new_labels])

	np.savetxt('./distance_matrix_with_classes.csv', new_matrix, delimiter=",", header=",".join(i for i in new_matrix.dtype.names), comments="", fmt='%s')
