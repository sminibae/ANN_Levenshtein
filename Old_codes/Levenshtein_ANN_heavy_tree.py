# Imports
from rapidfuzz.distance import Levenshtein
import numpy as np
import pandas as pd

import unicodedata
import random
import pickle
# import dill

from .ANN_Levenshtein.Template import TreeTemplate, IndexTemplate, ForestTemplate


class LevenshteinTree(TreeTemplate):
    def __init__(self, num_strings, split_num, depth, s1_idx=None, s2_idx=None, tf_array=None, weights=(1,1,1)):
        super().__init__(num_strings, split_num, depth, s1_idx, s2_idx, tf_array)
        self.weights = weights

    # Hangul Decomposition function using unicodedata
    def decompose_hangul(self, text, *args, **kwargs):
        decomposed = ''
        for char in text:
            if '가' <= char <= '힣':
                # Hangul syllables range: decompose
                decomposed += unicodedata.normalize('NFD', char)
            else:
                decomposed += char
        return decomposed
    
    def build_tree(self, strings, indices):
        # len(indices) == 0 or 1, can't split more
        if self.depth >= self.split_num or len(indices) <= 1:
            return None  # no node needed for empty/terminal group

        # len(indices) == 2, just split in two
        if len(indices) == 2:
            self.tf_array[indices[0], self.depth] = True
            self.tf_array[indices[1], self.depth] = False
            return LevenshteinTree(self.num_strings, self.split_num, self.depth, weights=self.weights)

        # Most cases
        rand_pos = np.random.choice(len(indices), size=2, replace=False)
        s1_idx = indices[rand_pos[0]]
        s2_idx = indices[rand_pos[1]]

        node = LevenshteinTree(self.num_strings, self.split_num, self.depth, s1_idx, s2_idx, self.tf_array, weights=self.weights)

        for idx in indices:
            d1 = Levenshtein.distance(strings[idx], strings[s1_idx], weights=self.weights, processor=self.decompose_hangul)
            d2 = Levenshtein.distance(strings[idx], strings[s2_idx], weights=self.weights, processor=self.decompose_hangul)
            self.tf_array[idx, self.depth] = (d1 >= d2)

        mask = self.tf_array[indices, self.depth]
        left = indices[np.flatnonzero(mask)]
        right = indices[np.flatnonzero(~mask)]

        node.left = LevenshteinTree(self.num_strings, self.split_num, self.depth + 1, tf_array=self.tf_array, weights=self.weights).build_tree(strings, left)
        node.right = LevenshteinTree(self.num_strings, self.split_num, self.depth + 1, tf_array=self.tf_array, weights=self.weights).build_tree(strings, right)

        return node

    def get_code(self, strings, new_str):
        fingerprint = np.zeros(self.split_num, dtype=bool)
        idx = 0
        node = self

        while node and node.s1_idx is not None and node.s2_idx is not None:
            d1 = Levenshtein.distance(new_str, strings[node.s1_idx], weights=self.weights, processor=self.decompose_hangul)
            d2 = Levenshtein.distance(new_str, strings[node.s2_idx], weights=self.weights, processor=self.decompose_hangul)
            go_left = d1 >= d2
            fingerprint[idx] = go_left
            node = node.left if go_left else node.right
            idx += 1
            
            # just in case
            assert idx < self.split_num, f"Fingerprint overflowed! idx={idx}, split_num={self.split_num}"


        return fingerprint

    def find_matches(self, strings, new_str, return_strings=False):
        fingerprint = self.get_code(strings, new_str)
        matches = np.where((self.tf_array == fingerprint).all(axis=1))[0]

        if return_strings:
            return matches, [strings[i] for i in matches]
        else:
            return matches
        
    def transform(self, strings: list[str]) -> np.ndarray:
        result = np.zeros((len(strings), self.split_num), dtype=bool)
        for i, s in enumerate(strings):
            result[i] = self.get_code(strings, s)
        return result

    def depth(self) -> int:
        def _max_depth(node):
            if node is None:
                return 0
            return 1 + max(_max_depth(node.left), _max_depth(node.right))

        return _max_depth(self)

    def save_tree(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_tree(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


class LevenshteinIndex(IndexTemplate):
    def __init__(self, num_trees, num_strings, split_num, weights=(1,1,1)):
        super().__init__(num_trees, num_strings, split_num)
        self._string_buffer = [None] * num_strings  # reserve space
        self._item_count = 0
        self.weights = weights
        
    # Hangul Decomposition function using unicodedata
    def decompose_hangul(self, text, *args, **kwargs):
        decomposed = ''
        for char in text:
            if '가' <= char <= '힣':
                # Hangul syllables range: decompose
                decomposed += unicodedata.normalize('NFD', char)
            else:
                decomposed += char
        return decomposed
    
    # add one-by-one
    def add_item(self, i: int, string: str):
        if self._string_buffer[i] is None:
            self._item_count += 1
        self._string_buffer[i] = string

    # add data at once
    def add_items_bulk(self, strings):
        if not isinstance(strings, (list, np.ndarray, pd.Series)):
            raise TypeError("Input must be a list, numpy array, or pandas Series of strings.")

        strings_array = np.asarray(strings, dtype=str)
        n = len(strings_array)

        if n > len(self._string_buffer):
            raise ValueError("Too many strings to add to the index.")

        # Update item count for previously None entries
        mask = np.array(self._string_buffer[:n], dtype=str) == None
        self._item_count += np.count_nonzero(mask)

        # Bulk assign to the internal buffer
        self._string_buffer[:n] = strings_array


    # Annoy style
    def build(self):
        strings = self._string_buffer
        for i in range(self.num_trees):
            root = LevenshteinTree(self.num_strings, self.split_num, 0, weights=self.weights)
            tree = root.build_tree(strings, np.arange(self.num_strings))
            self.trees.append(tree)

    def unbuild(self):
        self.trees = []
        return True
    

    def get_nns_by_vector(self, query_str, topk=None, include_distances=False):
        all_matches = []

        for tree in self.trees:
            match_indices = tree.find_matches(self._string_buffer, query_str)
            all_matches.extend(match_indices)

        if len(all_matches) == 0:
            return None
        
        # Remove duplicates, keep unique indices
        unique_matches = list(set(all_matches))

        if topk:
            scored = [(idx, self.get_distance_str(query_str, idx)) for idx in unique_matches]
            scored.sort(key=lambda x: x[1])
            result = scored[:topk]

            # Pad with None if needed
            if len(result) < topk:
                result += [(None, None)] * (topk - len(result))

            return result if include_distances else [idx for idx, _ in result]

        # if no topk specified
        if include_distances:
            return [(idx, self.get_distance_str(query_str, idx)) for idx in unique_matches]
        else:
            return unique_matches
    

    # same as get_nns_by_vector
    def get_nns_by_string(self, query_str, topk=None, include_distances=False):
        return self.get_nns_by_vector(query_str, topk, include_distances)

    def get_nns_by_item(self, i, n=None):
        query_str = self._string_buffer[i]
        return self.get_nns_by_string(query_str, topk=n)


    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    # def save(self, filename: str):
    #     with open(filename, "wb") as f:
    #         dill.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)
        
    # @staticmethod
    # def load(self, filename: str):
    #     with open(filename, "rb") as f:
    #         obj = dill.load(f)
    #     return obj
    
    def unload(self):
        self.trees = []
        self._string_buffer = []
        self._item_count = 0
        return True


    def on_disk_build(self, fn: str) -> bool:
        # Placeholder: no real on-disk building for Levenshtein trees
        print(f"[Warning] on_disk_build is not supported yet.")
        return True


    def get_distance(self, i: int, j: int) -> int:
        if not hasattr(self, '_string_buffer'):
            raise ValueError("No string data loaded. Use add_item() or build().")

        return Levenshtein.distance(self._string_buffer[i], self._string_buffer[j], weights=self.weights, processor=self.decompose_hangul)

    def get_distance_str(self, query_str: str, idx: int) -> int:
        return Levenshtein.distance(query_str, self._string_buffer[idx], weights=self.weights, processor=self.decompose_hangul)

    def get_n_items(self) -> int:
        return self._item_count
    
    def get_n_trees(self) -> int:
        return len(self.trees)
    
    def get_strings(self) -> list[str]:
        return self._string_buffer
    
    def get_item_vector(self, i: int, tree_id: int = 0) -> list[bool]:
        if not self.trees:
            raise ValueError("No trees built.")
        string = self._string_buffer[i]
        return self.trees[tree_id].get_code(self._string_buffer, string).tolist()
    

    def verbose(self, v: bool) -> bool:
        self._verbose = v
        return True

    def set_seed(self, s: int) -> None:
        random.seed(s)
        np.random.seed(s)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude the _string_buffer from being pickled
        state['_string_buffer'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Initialize empty buffer (or you can choose to load it later)
        self._string_buffer = [None] * self.num_strings

        

class LevenshteinForest(ForestTemplate):
    def __init__(self, n_estimators=10, max_depth=120, random_state=None, weights=(1,1,1)):
        super().__init__(n_estimators, max_depth, random_state)
        self.weights = weights
        
    def fit(self, strings: list[str]):
        self.strings = strings
        self.num_strings = len(strings)

        self.trees = []
        for _ in range(self.n_estimators):
            root = LevenshteinTree(self.num_strings, self.max_depth, 0, weights=self.weights)
            tree = root.build_tree(strings, np.arange(self.num_strings))
            self.trees.append(tree)

        self._is_fitted = True
        return self

    def transform(self, strings: list[str]) -> np.ndarray:
        assert self._is_fitted, "Call fit() before transform()"

        result = np.zeros((len(strings), self.n_estimators * self.max_depth), dtype=bool)

        for i, s in enumerate(strings):
            code_parts = []
            for tree in self.trees:
                code = tree.get_code(self.strings, s)
                code_parts.append(code)
            result[i] = np.concatenate(code_parts)

        return result
    
    def predict(self, queries: list[str], topk=1) -> list[list[str]]:
        assert self._is_fitted, "Call fit() before predict()"
        predictions = []

        for q in queries:
            match_counts = {}

            for tree in self.trees:
                indices = tree.find_matches(self.strings, q, return_strings=False)
                for idx in indices:
                    match_counts[idx] = match_counts.get(idx, 0) + 1

            # Sort matches by how often they appeared
            sorted_matches = sorted(match_counts.items(), key=lambda x: -x[1])
            top_indices = [idx for idx, _ in sorted_matches[:topk]]
            top_strings = [self.strings[i] for i in top_indices]

            predictions.append(top_strings)

        return predictions
    
    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    

    def __repr__(self):
        return (
            f"LevenshteinForest("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"random_state={self.random_state}, "
            f"is_fitted={self._is_fitted})"
        )
