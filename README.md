# protein_classification
A UMD CMSC701 project.
---------------------------------------------------------------------

**Test Command:**

*python NJTree.py {tree_data} {visualization_bool} {classification_method}*

**Examples with included dataset:**

*python NJTree.py protein_database True all*

*python NJTree.py protein_database True TreeNN*

*python NJTree.py protein_database False TreeInsert*

**To avoid building the tree, use pickled tree of the same dataset:**

*python NJTree.py ./data/protein_db_njtree.p True all*

The above command will work for any pickled NJTree you save.

