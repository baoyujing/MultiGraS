import os
import time
import json
import logging
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessors.utils import chunk_list, get_paths_of_root, get_paths_of_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self, stop_word_list=None, min_df=100, n_process=12):
        self.vectorizer = TfidfVectorizer(min_df=min_df, stop_words=stop_word_list, norm=None)
        self.n_process = n_process

    def fit(self, data_root, split_file):
        id2doc = self._load_data(data_root, split_file)
        self._get_tfidf(id2doc=id2doc)

    def _get_tfidf(self, id2doc):
        logger.info("Fitting data.")
        start_time = time.time()
        corpus = []   # sentences
        for doc in id2doc.values():
            corpus.extend(doc)
        logger.info("{:.2f}s elapsed.".format(time.time() - start_time))

    def build_adjacency_matrix_mp(self, data_root):
        logger.info("Building adjacency matrix.")
        start_time = time.time()

        file_names = get_paths_of_root(data_root=data_root)
        file_list = []
        for i, file_name in enumerate(file_names):
            if (i + 1) % 100000 == 0:
                logger.info("{} documents loaded, {:.2f}s elapsed.".format(i + 1, time.time() - start_time))
                self._mp_helper(file_list=file_list)
                file_list = []
            file_list.append(file_name)
        self._mp_helper(file_list=file_list)
        logger.info("{} documents processed, {:.2f}s elapsed.".format(len(file_names), time.time() - start_time))

    def _mp_helper(self, file_list):
        file_list_chunks = chunk_list(list(file_list), self.n_process)
        pool = Pool(self.n_process)

        input_list = []
        for i, file_chunk in enumerate(file_list_chunks):
            input_list.append((i, file_chunk))
        pool.map(self._create_graph_mp, input_list)

        pool.close()
        pool.join()

    def _create_graph_mp(self, inputs):
        worker_id, file_chunk = inputs
        start_time = time.time()
        for i, file_name in enumerate(file_chunk):
            if i % 1000 == 0:
                logger.info(
                    "Worker {}: {} documents processed, {:.2f}s elapsed.".format(
                        worker_id, i, time.time() - start_time))
            if not file_name.endswith("json"):
                continue
            data = json.load(open(file_name))
            graph = self._create_graph(document=data["document"])
            data["adj_matrix"] = graph
            json.dump(data, open(file_name, "w"))  # write back to the original file

        logger.info("Worker {}: {} total documents processed, {:.2f}s elapsed.".format(
            worker_id, len(file_chunk), time.time() - start_time))

    def _create_graph(self, document):
        document = [" ".join(d) for d in document]
        tfidf_mat = self.vectorizer.transform(document).toarray()
        adj_mat = cosine_similarity(tfidf_mat, tfidf_mat) + np.eye(len(tfidf_mat))   # tfidf is non-negative
        deg = np.power(np.sum(adj_mat, axis=1, keepdims=True), -0.5)   # [N, 1]
        adj_mat *= deg
        adj_mat *= deg.transpose()
        return adj_mat.tolist()

    def _load_data(self, data_root, split_file):
        logger.info("Loading {} data from {}.".format(split_file.split(".")[0], data_root))
        start_time = time.time()

        id2doc = {}
        file_names = get_paths_of_split(data_root=data_root, split_file=split_file)
        for i, file_name in enumerate(tqdm(file_names)):
            if not os.path.exists(file_name):  # skip the empty files
                continue
            data = json.load(open(file_name))
            document = data["document"]
            document = [" ".join(s) for s in document]
            id2doc[data["hashid"]] = document
        logger.info("{} documents loaded, {:.2f}s elapsed.".format(len(id2doc), time.time() - start_time))
        return id2doc
