import os
import hashlib
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_list(l, n_process):
    """
    Chunk a list into sub-lists.
    """
    chunk_size = int(np.ceil(len(l) / n_process))
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]


def get_paths_of_split(data_root, split_file):
    file_names = []
    with open(split_file) as f:
        for line in f:
            fn = os.path.join(data_root, line.strip())
            if not os.path.exists(fn):
                continue
            file_names.append(fn)
    return file_names


def get_paths_of_root(data_root):
    file_names = os.listdir(data_root)
    for i, fn in enumerate(file_names):
        file_names[i] = os.path.join(data_root, fn)
    return file_names


def hashhex(s):
    """
    Returns a heximal formated SHA1 hash of the input string.
    """
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()
