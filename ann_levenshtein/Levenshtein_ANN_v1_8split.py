# 2025.04.14
# improved from v6_parallel_tree_building
# splitting each node into 8 children
# with (insertion, deletion, substition) = Levenshtein.editops(s1,s2) score's plus/minus sign

# Imports
import numpy as np
import pandas as pd

import random
import unicodedata

from rapidfuzz.distance import Levenshtein
from .Template import IndexTemplate

from concurrent.futures import ThreadPoolExecutor
# from tqdm import tqdm


class LevenshteinIndex(IndexTemplate):
    def __init__(self, num_trees, n_jobs =-1):
        super().__init__(num_trees)
        # inputs
        self.num_trees = num_trees

        # Forest
        self.trees = []  
        # single tree = [tree_s1, tree_s2, 
        #         tree_ppp, tree_ppm, tree_pmp, tree_pmm, 
        #         tree_mpp, tree_mpm, tree_mmp, tree_mmm, 
        #         leaf_value]

        # data
        self._string_buffer = []
        self._item_count = 0

        # # options - build trees
        # self.max_depth = max_depth
        self.n_jobs = n_jobs

        # # options - calculating Levenshtine distance
        # self.weights = weights
        # self.use_processor = use_processor

    @staticmethod
    def Levenshtein_dist_vector(s1, s2):
        insertions, deletions, substitutions = 0, 0, 0
        for tag, _, _ in Levenshtein.editops(s1, s2):
            if tag == 'insert':
                insertions += 1
            elif tag == 'delete':
                deletions += 1
            elif tag == 'replace':
                substitutions += 1
        return (insertions, deletions, substitutions)
    
    # build each tree
    def _build_tree(self, strings, indices, tree_id):
        # Initialize
        estimated_nodes = 8 * len(self._string_buffer) + 8
        # if not max_depth:
        #     max_depth = len(self._string_buffer) + 1  # any integer bigger than data_len+1

        tree_s1 = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        tree_s2 = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        
        tree_ppp = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        tree_ppm = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        tree_pmp = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        tree_pmm = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        tree_mpp = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        tree_mpm = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        tree_mmp = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        tree_mmm = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)
        leaf_value = np.full(shape=estimated_nodes, fill_value=-1, dtype=np.int32)

        # queue
        queue = [(0, np.arange(len(self._string_buffer)))]  # (idx, indices)
        node_counter = 1

        # attempts
        # max_attempts = 50

        while queue:
            current_node_id, indices = queue.pop(0)

            # end as empty leaf
            if len(indices) == 0:
                continue

            # end as a leaf
            if len(indices) == 1:
                leaf_value[current_node_id] = indices[0]
                continue

            # # main split method with max_attempts
            # successful_split = False
            # # for _ in range(max_attempts):
            # while successful_split is not True:
                
            # Choose s1 and s2
            s1_idx, s2_idx = np.random.choice(indices, 2, replace=False)
            s1, s2 = strings[s1_idx], strings[s2_idx]
            if s1 == s2:
                continue

            # Compute Levenshtein distance
            mask_ppp = np.full(len(indices), False, dtype=bool)
            mask_ppm = np.full(len(indices), False, dtype=bool)
            mask_pmp = np.full(len(indices), False, dtype=bool)
            mask_pmm = np.full(len(indices), False, dtype=bool)
            mask_mpp = np.full(len(indices), False, dtype=bool)
            mask_mpm = np.full(len(indices), False, dtype=bool)
            mask_mmp = np.full(len(indices), False, dtype=bool)
            mask_mmm = np.full(len(indices), False, dtype=bool)
            
            for j, idx in enumerate(indices):
                each_string = strings[idx]
                (a1, a2, a3) = self.Levenshtein_dist_vector(each_string, s1)
                (b1, b2, b3) = self.Levenshtein_dist_vector(each_string, s2)
                # split into 8 parts
                if a1>=b1 and a2>=b2 and a3>=b3:  # +++
                    mask_ppp[j] = True  # go tree_ppp
                elif a1>=b1 and a2>=b2 and a3<b3:  # ++-
                    mask_ppm[j] = True  # go tree_ppm
                elif a1>=b1 and a2<b2 and a3>=b3:  # +-+
                    mask_pmp[j] = True  # go tree_pmp
                elif a1>=b1 and a2<b2 and a3<b3:  # +--
                    mask_pmm[j] = True  # go tree_pmm
                elif a1<b1 and a2>=b2 and a3>=b3:  # -++
                    mask_mpp[j] = True  # go tree_mpp
                elif a1<b1 and a2>=b2 and a3<b3:  # -+-
                    mask_mpm[j] = True  # go tree_mpm
                elif a1<b1 and a2<b2 and a3>=b3:  # --+
                    mask_mmp[j] = True  # go tree_mmp
                else:  # ---
                    mask_mmm[j] = True  # go tree_mmm
    
            indices_ppp = indices[mask_ppp]
            indices_ppm = indices[mask_ppm]
            indices_pmp = indices[mask_pmp]
            indices_pmm = indices[mask_pmm]
            indices_mpp = indices[mask_mpp]
            indices_mpm = indices[mask_mpm]
            indices_mmp = indices[mask_mmp]
            indices_mmm = indices[mask_mmm]

                # # at least 2 non-empty indices should exist
                # partitions = [
                #     indices_ppp, indices_ppm, indices_pmp, indices_pmm,
                #     indices_mpp, indices_mpm, indices_mmp, indices_mmm
                # ]
                # if sum(len(p) > 0 for p in partitions) >= 2:
                #     successful_split = True
                #     break  # found a good split

            # Assign s1, s2 idx
            tree_s1[current_node_id] = s1_idx
            tree_s2[current_node_id] = s2_idx

            # Safely get node idx for next left, right node
            id_ppp = node_counter
            id_ppm = node_counter + 1
            id_pmp = node_counter + 2
            id_pmm = node_counter + 3
            id_mpp = node_counter + 4
            id_mpm = node_counter + 5
            id_mmp = node_counter + 6
            id_mmm = node_counter + 7
            node_counter += 8

            # Assign next node idx for left, right node
            tree_ppp[current_node_id] = id_ppp
            tree_ppm[current_node_id] = id_ppm
            tree_pmp[current_node_id] = id_pmp
            tree_pmm[current_node_id] = id_pmm
            tree_mpp[current_node_id] = id_mpp
            tree_mpm[current_node_id] = id_mpm
            tree_mmp[current_node_id] = id_mmp
            tree_mmm[current_node_id] = id_mmm
            
            # Enqueue children
            queue.append((id_ppp, indices_ppp))
            queue.append((id_ppm, indices_ppm))
            queue.append((id_pmp, indices_pmp))
            queue.append((id_pmm, indices_pmm))
            queue.append((id_mpp, indices_mpp))
            queue.append((id_mpm, indices_mpm))
            queue.append((id_mmp, indices_mmp))
            queue.append((id_mmm, indices_mmm))

        tree = [
            tree_s1, tree_s2, 
            tree_ppp, tree_ppm, tree_pmp, tree_pmm, 
            tree_mpp, tree_mpm, tree_mmp, tree_mmm, 
            leaf_value
        ]

        return tree
    
    # find match for query string using tree
    def _query(self, tree, query_str):
        (tree_s1, tree_s2, 
         tree_ppp, tree_ppm, tree_pmp, tree_pmm, 
         tree_mpp, tree_mpm, tree_mmp, tree_mmm, 
         leaf_value) = tree
        data = self._string_buffer

        idx = 0  # root node
        while True:
            # leaf with a value
            if leaf_value[idx] != -1:
                return leaf_value[idx]  # returns idx of data, closest match for this tree
            
            # Dead-end: no children & not a leaf
            if all(child[idx] == -1 for child in [
                tree_ppp, tree_ppm, tree_pmp, tree_pmm,
                tree_mpp, tree_mpm, tree_mmp, tree_mmm
                ]) and leaf_value[idx] == -1:
                return None  # returns None, no match for this tree
            
            # keep go into tree
            # Compute Levenshtein distance vector
            s1_idx, s2_idx = tree_s1[idx], tree_s2[idx]
            s1, s2 = data[s1_idx], data[s2_idx]
            (a1, a2, a3) = self.Levenshtein_dist_vector(query_str, s1)
            (b1, b2, b3) = self.Levenshtein_dist_vector(query_str, s2)

            # Traverse based on comparison
            if a1>=b1 and a2>=b2 and a3>=b3:  # +++
                idx = tree_ppp[idx]  # go tree_ppp
            elif a1>=b1 and a2>=b2 and a3<b3:  # ++-
                idx = tree_ppm[idx]  # go tree_ppm
            elif a1>=b1 and a2<b2 and a3>=b3:  # +-+
                idx = tree_pmp[idx]  # go tree_pmp
            elif a1>=b1 and a2<b2 and a3<b3:  # +--
                idx = tree_pmm[idx]  # go tree_pmm
            elif a1<b1 and a2>=b2 and a3>=b3:  # -++
                idx = tree_mpp[idx]  # go tree_mpp
            elif a1<b1 and a2>=b2 and a3<b3:  # -+-
                idx = tree_mpm[idx]  # go tree_mpm
            elif a1<b1 and a2<b2 and a3>=b3:  # --+
                idx = tree_mmp[idx]  # go tree_mmp
            else:  # ---
                idx = tree_mmm[idx]  # go tree_mmm

            if idx == -1:
                return None  # prevent crash


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
    
    # def build(self):
    #     for _ in range(self.num_trees):
    #         tree = self._build_tree(self._string_buffer, np.arange(len(self._string_buffer)))
    #         self.trees.append(tree)


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
            if match_idx is not None:
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
        # result = Levenshtein.distance(self._string_buffer[i], self._string_buffer[j], weights=self.weights, processor=self._decompose_korean)
        result = Levenshtein.distance(self._string_buffer[i], self._string_buffer[j])
        return result

    def get_distance_str(self, query_str: str, idx: int) -> int:
            # result = Levenshtein.distance(query_str, self._string_buffer[idx], weights=self.weights, processor=self._decompose_korean)
        result = Levenshtein.distance(query_str, self._string_buffer[idx])
        return result


    def save(self, filename: str): # *.npz
        arrays = {}
        for i, tree in enumerate(self.trees):
            arrays[f'tree_{i}_s1'] = tree[0]
            arrays[f'tree_{i}_s2'] = tree[1]

            arrays[f'tree_{i}_ppp'] = tree[2]
            arrays[f'tree_{i}_ppm'] = tree[3]
            arrays[f'tree_{i}_pmp'] = tree[4]
            arrays[f'tree_{i}_pmm'] = tree[5]
            
            arrays[f'tree_{i}_mpp'] = tree[6]
            arrays[f'tree_{i}_mpm'] = tree[7]
            arrays[f'tree_{i}_mmp'] = tree[8]
            arrays[f'tree_{i}_mmm'] = tree[9]

            arrays[f'tree_{i}_leaf'] = tree[10]

        np.savez_compressed(filename, **arrays)

    def load(self, filename: str): # *.npz
        num_trees = self.num_trees
        self.trees = []

        data = np.load(filename)
        for i in range(num_trees):
            tree = [
                data[f'tree_{i}_s1'],
                data[f'tree_{i}_s2'],

                data[f'tree_{i}_ppp'],
                data[f'tree_{i}_ppm'],
                data[f'tree_{i}_pmp'],
                data[f'tree_{i}_pmm'],

                data[f'tree_{i}_mpp'],
                data[f'tree_{i}_mpm'],
                data[f'tree_{i}_mmp'],
                data[f'tree_{i}_mmm'],
                
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
    