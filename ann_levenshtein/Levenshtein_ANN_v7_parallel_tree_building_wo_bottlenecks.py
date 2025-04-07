# Imports
import numpy as np
import pandas as pd

import random
import unicodedata

from rapidfuzz.distance.Levenshtein import distance as dist
from rapidfuzz.process import cdist
from .Template import IndexTemplate

from collections import deque
from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm


class LevenshteinIndex(IndexTemplate):
    def __init__(self, num_trees, n_jobs =-1):
        super().__init__(num_trees)
        # inputs
        self.num_trees = num_trees

        # Forest
        self.trees = []  
        # single tree = [tree_s1, tree_s2, tree_left, tree_right, leaf_value]

        # data
        self._string_buffer = []
        self._item_count = 0

        # # options - build trees
        # self.max_depth = max_depth
        self.n_jobs = n_jobs

        # # options - calculating Levenshtine distance
        # self.weights = weights
        # self.use_processor = use_processor


    # Korean decomposition function using unicodedata
    def _decompose_korean(text, *args, **kwargs):
        decomposed = ''
        for char in text:
            if '가' <= char <= '힣':
                # Hangul syllables range: decompose
                decomposed += unicodedata.normalize('NFD', char)
            else:
                decomposed += char
        return decomposed
    

    # build each tree
    def _build_tree(self, strings, indices, tree_id):
        # Initialize
        estimated_nodes = 2 * len(self._string_buffer) + 2
        # if not max_depth:
        #     max_depth = len(self._string_buffer) + 1  # any integer bigger than data_len+1

        tree_s1 = np.zeros(shape=estimated_nodes, dtype=np.int32)
        tree_s2 = np.zeros(shape=estimated_nodes, dtype=np.int32)
        tree_left = np.zeros(shape=estimated_nodes, dtype=np.int32)
        tree_right = np.zeros(shape=estimated_nodes, dtype=np.int32)
        leaf_value = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)

        # queue
        queue = deque()
        queue.append((0, np.arange(len(self._string_buffer))))  # (idx, indices)
        node_counter = 1

        # attempts
        max_attempts = 5

        # progress bar for queue
        # pbar_queue = tqdm(total=estimated_nodes, position=tree_id*3, desc=f"Tree {tree_id} queue", leave=False)
        # progress bar for total calculations
        # distance_counter = 0
        # pbar_total_dists = tqdm(total=0, position=tree_id,  bar_format='{desc}: {n_fmt}', desc=f"Tree {tree_id} total dists", leave=False, unit_scale=True)

        while queue:
            current_node_id, indices = queue.popleft()  # O(1)

            # update pbar
            # pbar_queue.update(1)
            # pbar_queue.refresh()
            
            # end as leaf
            # if depth >= max_depth or len(indices) <= 1:  
            if len(indices) <= 1:  
                leaf_value[current_node_id] = indices[0]
                tree_left[current_node_id] = -1
                tree_right[current_node_id] = -1
                continue

            # main split method with max_attempts
            successful_split = False
            for _ in range(max_attempts):
                # Choose s1 and s2
                s1_idx, s2_idx = np.random.choice(indices, 2, replace=False)
                s1, s2 = strings[s1_idx], strings[s2_idx]
                if s1 == s2:
                    continue

                # Compute Levenshtein distance
                mask = np.empty(len(indices), dtype=bool)

                subset = strings[indices]
                # Vectorized distance calculation (parallelized with all cores)
                d1s = cdist([s1], subset, scorer=dist, dtype=np.int32, workers=-1)[0]
                d2s = cdist([s2], subset, scorer=dist, dtype=np.int32, workers=-1)[0]

                # Build the mask as before
                mask = d1s <= d2s

                # # progress bar for calculating Levenshtein distances
                # # with tqdm(total=len(indices), position=tree_id*3+2, desc=f"Tree {tree_id} distances", leave=False) as pbar_dist:
                # for j, idx in enumerate(indices):
                #     d1 = dist(strings[idx], s1)
                #     d2 = dist(strings[idx], s2)
                #     # d1 <= d2, go left, True
                #     mask[j] = d1 <= d2
                #         # pbar_dist.update(1)
                
                # counting total dist calculations
                # distance_counter += len(indices)
                # pbar_total_dists.n = distance_counter
                # pbar_total_dists.refresh()

                left_indices = indices[mask]
                right_indices = indices[~mask]

                if len(left_indices) > 0 and len(right_indices) > 0:
                    successful_split = True
                    break  # found a good split

            # If split failed after attempts, fallback to manual split
            if not successful_split:
                mid = len(indices) // 2
                left_indices = indices[:mid]
                right_indices = indices[mid:]
                s1_idx, s2_idx = indices[0], indices[-1]

            # Assign s1, s2 idx
            tree_s1[current_node_id] = s1_idx
            tree_s2[current_node_id] = s2_idx

            # Safely get node idx for next left, right node
            left_id = node_counter
            right_id = node_counter + 1
            node_counter += 2

            # Assign next node idx for left, right node
            tree_left[current_node_id] = left_id
            tree_right[current_node_id] = right_id

            # Enqueue children
            queue.append((left_id, left_indices))
            queue.append((right_id, right_indices)) 

        tree = [tree_s1, tree_s2, tree_left, tree_right, leaf_value]
        
        # close pbars
        # pbar_queue.close()
        # pbar_total_dists.close()

        return tree
    
    # find match for query string using tree
    def _query(self, tree, query_str):
        tree_s1, tree_s2, tree_left, tree_right, leaf_value = tree
        data = self._string_buffer

        idx = 0  # root node
        while True:
            if tree_left[idx] == -1 and tree_right[idx] == -1:
                return leaf_value[idx]  # returns idx of data, closest match for this tree
            
            s1_idx = tree_s1[idx]
            s2_idx = tree_s2[idx]
            d1 = dist(query_str, data[s1_idx])
            d2 = dist(query_str, data[s2_idx])

            idx = tree_left[idx] if d1 <= d2 else tree_right[idx]


    def add_item(self, i: int, string: str):
        self._item_count += 1
        self._string_buffer.append(string)

    def add_items_bulk(self, strings):
        if not isinstance(strings, (list, np.ndarray, pd.Series)):
            raise TypeError("Input must be a list, numpy array, or pandas Series of strings.")

        self._string_buffer = np.asarray(strings, dtype=str)

    def unload(self):
        self._string_buffer = None
        return True
    
    def build(self):
        for _ in range(self.num_trees):
            tree = self._build_tree(self._string_buffer, np.arange(len(self._string_buffer)))
            self.trees.append(tree)


    def build(self):
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [
                executor.submit(
                    self._build_tree, 
                    self._string_buffer, 
                    np.arange(len(self._string_buffer)),
                    tree_id
                )
                for tree_id in range(self.num_trees)
            ]
            for future in futures:
                self.trees.append(future.result())

    def unbuild(self):
        self.trees = []
        return True
    
    def get_nns_by_vector(self, query_str, topk=None, include_distances=False):
        all_matches = []
        for tree in self.trees:
            match_idx = self._query(tree, query_str)
            all_matches.append(match_idx)

        if not all_matches:
            return [] if not include_distances else ([], [])

        # Deduplicate matches
        unique_matches = list(set(all_matches))

        # Score them by Levenshtein distance
        scored = [(idx, self.get_distance_str(query_str, idx)) for idx in unique_matches]
        scored.sort(key=lambda x: x[1])  # sort by distance

        # Take top-k
        if topk:
            scored = scored[:topk]

        # Pad if not enough results
        while len(scored) < topk:
            scored.append((None, float('inf')))

        # Return based on include_distances flag
        if include_distances:
            indices, distances = zip(*scored)
            return list(indices), list(distances)
        else:
            return [idx for idx, _ in scored]
        
    def get_nns_by_string(self, query_str, topk=None, include_distances=False):
        return self.get_nns_by_vector(query_str, topk, include_distances)

    def get_nns_by_item(self, i, topk=None, include_distances=False):
        query_str = self._string_buffer[i]
        return self.get_nns_by_vector(query_str, topk, include_distances)
    
    def get_item_vector(self, i: int, tree_id: int = 0) -> str:
        return self._string_buffer[i]
    
    def get_distance(self, i: int, j: int) -> int:
        # result = dist(self._string_buffer[i], self._string_buffer[j], weights=self.weights, processor=self._decompose_korean)
        result = dist(self._string_buffer[i], self._string_buffer[j])
        return result

    def get_distance_str(self, query_str: str, idx: int) -> int:
            # result = dist(query_str, self._string_buffer[idx], weights=self.weights, processor=self._decompose_korean)
        result = dist(query_str, self._string_buffer[idx])
        return result


    def save(self, filename: str): # *.npz
        arrays = {}
        for i, tree in enumerate(self.trees):
            arrays[f'tree_{i}_s1'] = tree[0]
            arrays[f'tree_{i}_s2'] = tree[1]
            arrays[f'tree_{i}_left'] = tree[2]
            arrays[f'tree_{i}_right'] = tree[3]
            arrays[f'tree_{i}_leaf'] = tree[4]

        np.savez_compressed(filename, **arrays)

    def load(self, filename: str): # *.npz
        num_trees = self.num_trees
        self.trees = []

        data = np.load(filename)
        for i in range(num_trees):
            tree = [
                data[f'tree_{i}_s1'],
                data[f'tree_{i}_s2'],
                data[f'tree_{i}_left'],
                data[f'tree_{i}_right'],
                data[f'tree_{i}_leaf']
            ]
            self.trees.append(tree)

        
    def get_n_items(self) -> int:
        return self._item_count

    def get_n_trees(self) -> int:
        return len(self.trees)

    def get_strings(self) -> list[str]:
        return self._string_buffer


    def set_seed(self, s: int) -> None:
        random.seed(s)
        np.random.seed(s)

    def verbose(self, v: bool) -> bool:
        self._verbose = v
        return True

    def on_disk_build(self, fn: str) -> bool:
        print(f"[Warning] on_disk_build is not supported yet.")
        return True
    