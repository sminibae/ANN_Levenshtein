import numpy as np
import pandas as pd 

from ann_levenshtein import LevenshteinIndex

def main():
    data = pd.read_csv('Data/IMK_master.txt', delimiter='\t', dtype=str)
    data['label'] = data['label'].astype(str)
    data['description'] = data['description'].astype(str)

    leven_index = LevenshteinIndex(num_trees=100, n_jobs=20)
    leven_index.add_items_bulk(data['description'])

    print('start')
    leven_index.build()

    leven_index.save("Models/LevenshteinIndex_IMK_master_100tree_v7.npz")
    
if __name__ == "__main__":
    main()