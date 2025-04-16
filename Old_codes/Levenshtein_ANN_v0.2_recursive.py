# Imports
import numpy as np
import pandas as pd

import random
import pickle
import joblib
# import dill
import unicodedata

from rapidfuzz.distance import Levenshtein
from .Template import IndexTemplate


# class LevenshteinNode(NodeTemplate):
#     def __init__(self, s1_idx=None, s2_idx=None, depth=0):
#         super().__init__(s1_idx, s2_idx, depth)
#         self.s1_idx = s1_idx
#         self.s2_idx = s2_idx
#         self.depth = depth
#         self.left = None
#         self.right = None



class LevenshteinIndex(IndexTemplate):
    def __init__(self, num_trees, num_strings, split_num, weights=(1, 1, 1), use_processor=False):
        super().__init__(num_trees, num_strings, split_num)
        # inputs
        self.num_trees = num_trees
        self.num_strings = num_strings
        self.split_num = split_num

        # main index
        self.trees = []
        self.tree_maps = []
        self.tf_array = np.zeros((num_strings, split_num), dtype=bool)

        # data buffer
        self._string_buffer = [None] * num_strings
        self._item_count = 0

        # options
        self.weights = weights
        self.use_processor = use_processor


    def _decompose_korean(self, text):
        return ''.join(unicodedata.normalize('NFD', ch) if '가' <= ch <= '힣' else ch for ch in text)

    def _build_tree(self, strings, indices, tree=None, tree_map=None, depth=0, node_idx=1):
        # start
        if tree is None:
            tree = []
        if tree_map is None:
            tree_map = {}

        # can't split more
        if depth >= self.split_num or len(indices) <= 1:
            return  

        # 2 strings left, just split in two
        if len(indices) == 2:  
            self.tf_array[indices[0], depth] = True
            self.tf_array[indices[1], depth] = False
            # node = [node_idx, s1_idx, s2_idx]
            node = [node_idx, indices[0], indices[1]]
            tree.append(node)
            # tree_map = {node_idx: (s1_idx, s2_idx)}
            tree_map[node_idx] = (indices[0], indices[1])
            return

        s1_idx, s2_idx = np.random.choice(indices, 2, replace=False)
        node = [node_idx, s1_idx, s2_idx]
        tree.append(node)
        tree_map[node_idx] = (s1_idx, s2_idx)

        mask = []
        for idx in indices:
            if self.use_processor==True:
                d1 = Levenshtein.distance(strings[idx], strings[s1_idx], weights=self.weights, processor=self._decompose)
                d2 = Levenshtein.distance(strings[idx], strings[s2_idx], weights=self.weights, processor=self._decompose) 
            else:
                d1 = Levenshtein.distance(strings[idx], strings[s1_idx], weights=self.weights)
                d2 = Levenshtein.distance(strings[idx], strings[s2_idx], weights=self.weights)
            # close to s1, True. go left
            self.tf_array[idx, depth] = (d1 <= d2)
            mask.append(d1 <= d2)

        mask = np.array(mask)
        left_indices = indices[mask]
        right_indices = indices[~mask]

        # node.left
        self._build_tree(strings, left_indices, tree, tree_map, depth + 1, 2*node_idx)
        # node.right
        self._build_tree(strings, right_indices, tree, tree_map, depth + 1, 2*node_idx+1)

        # moved to self.build() for not sorting at every level of recursion.
        # tree = np.array(tree, dtype=np.int32)
        # tree = tree[np.argsort(tree[:, 0])]  # sort by node_idx

        return tree, tree_map

    def _get_code(self, tree_map, new_str):
        fingerprint = np.zeros(self.split_num, dtype=bool)
        node_idx = 1
        depth = 0

        # Assume tree is sorted by node_idx and indexed from 1
        # tree_map = {node_idx: (s1_idx, s2_idx)}
        while True:
        
            split_vectors = tree_map.get(node_idx)
            if split_vectors is None:
                break  # reached leaf or missing node

            s1_idx, s2_idx = split_vectors

            if self.use_processor == True:
                d1 = Levenshtein.distance(new_str, self._string_buffer[s1_idx], weights=self.weights, processor=self._decompose)
                d2 = Levenshtein.distance(new_str, self._string_buffer[s2_idx], weights=self.weights, processor=self._decompose)
            else:
                d1 = Levenshtein.distance(new_str, self._string_buffer[s1_idx], weights=self.weights)
                d2 = Levenshtein.distance(new_str, self._string_buffer[s2_idx], weights=self.weights)

            # close to s1, True. go left
            go_left = (d1 <= d2)
            fingerprint[depth] = go_left
            node_idx = 2*node_idx if go_left else 2*node_idx+1
            depth += 1

            if depth >= self.split_num:
                raise RuntimeError("Fingerprint overflowed.")
            
        return fingerprint
    
    def _depth(self, node) -> int:
        def _max_depth(node):
            if node is None:
                return 0
            return 1 + max(_max_depth(node.left), _max_depth(node.right))

        return _max_depth(node)
    
    def add_item(self, i: int, string: str):
        if self._string_buffer[i] is None:
            self._item_count += 1
        self._string_buffer[i] = string

    def add_items_bulk(self, strings):
        if not isinstance(strings, (list, np.ndarray, pd.Series)):
            raise TypeError("Input must be a list, numpy array, or pandas Series of strings.")

        strings_array = np.asarray(strings, dtype=str)
        n = len(strings_array)

        if n > len(self._string_buffer):
            raise ValueError("Too many strings to add to the index.")

        mask = np.array(self._string_buffer[:n], dtype=object) == None
        self._item_count += np.count_nonzero(mask)

        self._string_buffer[:n] = strings_array

    def build(self):
        for _ in range(self.num_trees):
            tree, tree_map = self._build_tree(self._string_buffer, np.arange(self.num_strings))
            tree = np.array(tree, dtype=np.int32)
            tree = tree[np.argsort(tree[:, 0])]  # sort by node_idx
            self.trees.append(tree)
            self.tree_maps.append(tree_map)

    def unbuild(self):
        self.trees = []
        self.tree_maps = []
        return True

    def transform(self, strings):
        result = np.zeros((len(strings), self.split_num), dtype=bool)
        for i, s in enumerate(strings):
            result[i] = self._get_code(self.trees[0], s)
        return result
j
    def get_nns_by_vector(self, query_str, topk=None, include_distances=False):
        all_matches = []

        for tree in self.trees:
            query_fp = self._get_code(tree, query_str)
            matches = np.where((self.tf_array == query_fp).all(axis=1))[0]
            all_matches.extend(matches)

        if not all_matches:
            return None

        unique_matches = list(set(all_matches))

        if topk:
            scored = [(idx, self.get_distance_str(query_str, idx)) for idx in unique_matches]
            scored.sort(key=lambda x: x[1])
            result = scored[:topk]

            if len(result) < topk:
                result += [(None, None)] * (topk - len(result))

            return result if include_distances else [idx for idx, _ in result]

        return [(idx, self.get_distance_str(query_str, idx)) for idx in unique_matches] if include_distances else unique_matches

    def get_nns_by_string(self, query_str, topk=None, include_distances=False):
        return self.get_nns_by_vector(query_str, topk, include_distances)

    def get_nns_by_item(self, i, topk=None, include_distances=False):
        query_str = self._string_buffer[i]
        return self.get_nns_by_vector(query_str, topk, include_distances)

    def get_distance(self, i: int, j: int) -> int:
        if self.use_processor == True:
            result = Levenshtein.distance(self._string_buffer[i], self._string_buffer[j], weights=self.weights, processor=self._decompose)
        else:
            result = Levenshtein.distance(self._string_buffer[i], self._string_buffer[j], weights=self.weights)
        return result

    def get_distance_str(self, query_str: str, idx: int) -> int:
        if self.use_processor == True:
            result = Levenshtein.distance(query_str, self._string_buffer[idx], weights=self.weights, processor=self._decompose)
        else:
            result = Levenshtein.distance(query_str, self._string_buffer[idx], weights=self.weights)
        return result

    def save(self, filename: str, use_joblib=True, use_pickle=False):
        if use_joblib:
            with open(filename+'trees.joblib', 'wb') as f:
                joblib.dump(self.trees, f)
            with open(filename+'tf_array.joblib', 'wb') as f:
                joblib.dump(self.tf_array, f)
            with open(filename+'tree_maps.joblib', 'wb') as f:
                joblib.dump(self.tree_maps, f)
                
        elif use_pickle:
            with open(filename+'trees.pkl', "wb") as f:
                pickle.dump(self.trees, f)
            with open(filename+'tf_array.pkl', 'wb') as f:
                pickle.dump(self.tf_array, f)
            with open(filename+'tree_maps.pkl', 'wb') as f:
                pickle.dump(self.tree_maps, f)
                
        else:
           raise ValueError("Must specify at least one method: use_joblib or use_pickle.")

    def load(self, filename: str, use_joblib=True, use_pickle=False):
        if use_joblib:
            with open(filename + 'trees.joblib', 'rb') as f:
                self.trees = joblib.load(f)
            with open(filename + 'tf_array.joblib', 'rb') as f:
                self.tf_array = joblib.load(f)
            with open(filename + 'tree_maps.joblib', 'rb') as f:
                self.tree_maps = joblib.load(f)

        elif use_pickle:
            with open(filename + 'trees.pkl', 'rb') as f:
                self.trees = pickle.load(f)
            with open(filename + 'tf_array.pkl', 'rb') as f:
                self.tf_array = pickle.load(f)
            with open(filename + 'tree_maps.pkl', 'rb') as f:
                self.tree_maps = pickle.load(f)
        else:
            raise ValueError("Must specify at least one method: use_joblib or use_pickle.")

        
    def unload(self):
        self.trees = []
        self.tree_maps = []
        self.tf_array = None
        return True

    def get_n_items(self) -> int:
        return self._item_count

    def get_n_trees(self) -> int:
        return len(self.trees)

    def get_strings(self) -> list[str]:
        return self._string_buffer

    def get_item_vector(self, i: int, tree_id: int = 0) -> list[bool]:
        if not self.trees:
            raise ValueError("No trees built.")
        return self._get_code(self.trees[tree_id], self._string_buffer[i]).tolist()

    def set_seed(self, s: int) -> None:
        random.seed(s)
        np.random.seed(s)

    def verbose(self, v: bool) -> bool:
        self._verbose = v
        return True

    def on_disk_build(self, fn: str) -> bool:
        print(f"[Warning] on_disk_build is not supported yet.")
        return True
    
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state['_string_buffer'] = None
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     self._string_buffer = [None] * self.num_strings