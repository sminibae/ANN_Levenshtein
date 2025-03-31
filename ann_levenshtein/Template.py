# Imports
import numpy as np

from abc import ABC, abstractmethod
import random


class NodeTemplate(ABC):
    def __init__(self, s1_idx=None, s2_idx=None, depth=0):
        self.s1_idx = s1_idx
        self.s2_idx = s2_idx
        self.depth = depth
        self.left = None
        self.right = None


class IndexTemplate(ABC):
    def __init__(self, num_trees, num_strings, split_num):
        self.num_trees = num_trees
        self.num_strings = num_strings
        self.split_num = split_num
        self.trees = []

    @abstractmethod
    def _build_tree(self, strings, indices, depth):
        pass

    @abstractmethod       
    def add_item(self, i: int, string: str):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def unbuild(self):
        pass

    @abstractmethod
    def get_nns_by_vector(self, query_str, topk=None, include_distances=False):
        pass

    @abstractmethod
    def get_nns_by_item(self, i, topk=None, include_distances=False):
        pass
    
    @abstractmethod
    def save(self, filename: str):
        pass

    @staticmethod
    @abstractmethod
    def load(filename: str):
        pass

    @abstractmethod
    def unload(self):
        pass

    @abstractmethod
    def on_disk_build(self, fn: str) -> bool:
        pass
    

class TreeTemplate(ABC):
    def __init__(self, num_strings, split_num, depth, s1_idx=None, s2_idx=None, tf_array=None):
        self.num_strings = num_strings
        self.split_num = split_num
        self.depth = depth
        self.s1_idx = s1_idx
        self.s2_idx = s2_idx
        self.left = None
        self.right = None

        if tf_array is None and depth == 0:
            self.tf_array = np.zeros((num_strings, split_num), dtype=bool)
        else:
            self.tf_array = tf_array

    @abstractmethod
    def build_tree(self, strings, indices):
        pass

    @abstractmethod
    def find_matches(self, strings, new_str):
        pass
    
    @abstractmethod
    def save_tree(self, filename):
        pass

    @staticmethod
    @abstractmethod
    def load_tree(filename):
        pass


class ForestTemplate(ABC):
    def __init__(self, n_estimators, max_depth, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.strings = []
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
        self._is_fitted = False 


    @abstractmethod
    def fit(self, strings: list[str]):
        pass

    @abstractmethod
    def transform(self, strings: list[str]) -> np.ndarray:
        pass

    @abstractmethod
    def predict(self, queries: list[str], topk=1) -> list[list[str]]:
        pass
    
    @abstractmethod
    def save(self, filename: str):
        pass

    @staticmethod
    @abstractmethod
    def load(filename: str):
        pass

    @abstractmethod
    def get_params(self, deep=True):
        pass

    @abstractmethod
    def set_params(self, **params):
        pass

    @abstractmethod
    def __repr__(self):
        pass
      