'''
Rosalind: Contruct a Trie from a Collection of Patterns

Tara Larrue
CMSC701
'''
import sys


class Trie:
	def __init__(self):
		self.activeNode = 0
		self.maxNode = 0
		self.trie = {0:{}} #nested dictionaries - {startnode#: {symbol:endnode..} }

	def getSymbolsFromActiveNode(self):
		return self.trie[self.activeNode].keys()

	def goToRoot(self):
		self.activeNode = 0

	def goToSymbol(self, symbol):
		self.activeNode = self.trie[self.activeNode][symbol]

	def addNewNode(self, symbol):
		self.maxNode += 1
		self.trie[self.activeNode][symbol] = self.maxNode #add edge
		self.trie[self.maxNode] = {} #add node
		self.activeNode = self.maxNode

	def toAdjacentList(self, filename):
		f = open(filename, 'w')

		for startnode,edge in self.trie.items():
			for symbol, endnode in edge.items():
				f.write(str(startnode) + "->" + str(endnode) + ":" + symbol+"\n")

		f.close()

		print "New File Saved: ", filename

def trieConstruction(patterns, outputfile):
	'''
	Input: list of strings
	Output: Representation of a Trie in form of an adjacency list
	'''

	myTrie = Trie()

	for pattern in patterns:

		myTrie.goToRoot()

		for letter in pattern:

			if letter in myTrie.getSymbolsFromActiveNode():

				myTrie.goToSymbol(letter)

			else:

				myTrie.addNewNode(letter)

	myTrie.toAdjacentList(outputfile)

	return None

def main(inputfile, outputfile):

	txt = open(inputfile, 'r')

	patterns = []
	for line in txt:
		patterns.append(line.strip(' \n').upper())

	txt.close()

	trieConstruction(patterns, outputfile)


if __name__=='__main__':
	inputfile = sys.argv[1]
	outputfile = sys.argv[2]
	main(inputfile, outputfile)




