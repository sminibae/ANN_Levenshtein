# Imports
import numpy as np
import pandas as pd

import random
import pickle
# import dill
import unicodedata

from rapidfuzz.distance import Levenshtein
from .Template import NodeTemplate, IndexTemplate, TreeTemplate, ForestTemplate


class LevenshteinNode(NodeTemplate):
    def __init__(self, s1_idx=None, s2_idx=None, depth=0):
        super().__init__(s1_idx, s2_idx, depth)


class LevenshteinIndex(IndexTemplate):
    def __init__(self, num_trees, num_strings, split_num, weights=(1, 1, 1)):
        super().__init__(num_trees, num_strings, split_num)
        self.weights = weights
        self._string_buffer = [None] * num_strings
        self._item_count = 0
        self.trees = []

    def _decompose_korean(self, text):
        return ''.join(unicodedata.normalize('NFD', ch) if '가' <= ch <= '힣' else ch for ch in text)

    def _build_tree(self, strings, indices, depth):
        if depth >= self.split_num or len(indices) <= 1:  # len(indices) == 0 or 1, can't split more
            return None  # no node needed for empty/terminal group

        if len(indices) == 2:  # len(indices) == 2, just split in two
            return LevenshteinNode(indices[0], indices[1], depth)

        s1_idx, s2_idx = np.random.choice(indices, 2, replace=False)
        node = LevenshteinNode(s1_idx, s2_idx, depth)

        mask = []
        for idx in indices:
            d1 = Levenshtein.distance(strings[idx], strings[s1_idx], weights=self.weights, processor=self._decompose)
            d2 = Levenshtein.distance(strings[idx], strings[s2_idx], weights=self.weights, processor=self._decompose)
            mask.append(d1 >= d2)

        mask = np.array(mask)
        left_indices = indices[~mask]
        right_indices = indices[mask]

        node.left = self._build_tree(strings, left_indices, depth + 1)
        node.right = self._build_tree(strings, right_indices, depth + 1)

        return node

    def _get_code(self, node, new_str):
        fingerprint = np.zeros(self.split_num, dtype=bool)
        idx = 0

        while node and node.s1_idx is not None and node.s2_idx is not None:
            d1 = Levenshtein.distance(new_str, self._string_buffer[node.s1_idx], weights=self.weights, processor=self._decompose)
            d2 = Levenshtein.distance(new_str, self._string_buffer[node.s2_idx], weights=self.weights, processor=self._decompose)
            go_left = d1 >= d2
            fingerprint[idx] = go_left
            node = node.left if go_left else node.right
            idx += 1

            if idx >= self.split_num:
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
            tree = self._build_tree(self._string_buffer, np.arange(self.num_strings), 0)
            self.trees.append(tree)

    def unbuild(self):
        self.trees = []
        return True

    def transform(self, strings):
        result = np.zeros((len(strings), self.split_num), dtype=bool)
        for i, s in enumerate(strings):
            result[i] = self._get_code(self.trees[0], s)
        return result

    def get_nns_by_vector(self, query_str, topk=None, include_distances=False):
        all_matches = []

        for tree in self.trees:
            query_fp = self._get_code(tree, query_str)
            for i in range(self.num_strings):
                item_fp = self._get_code(tree, self._string_buffer[i])
                if np.array_equal(query_fp, item_fp):
                    all_matches.append(i)

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
        return Levenshtein.distance(self._string_buffer[i], self._string_buffer[j], weights=self.weights, processor=self._decompose)

    def get_distance_str(self, query_str: str, idx: int) -> int:
        return Levenshtein.distance(query_str, self._string_buffer[idx], weights=self.weights, processor=self._decompose)

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def unload(self):
        self.trees = []
        self._string_buffer = []
        self._item_count = 0
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
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_string_buffer'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._string_buffer = [None] * self.num_strings