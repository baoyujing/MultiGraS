import torch
import logging
import numpy as np

from data_management.vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyEmbedding:
    def __init__(self, configs, vocabulary):
        self.configs = self.default_configs()
        self.configs.update(configs)

        self.vocabulary = vocabulary
        self.word_emb_pretrained = torch.nn.Embedding(len(vocabulary), self.configs["dim_emb"], padding_idx=0)

        emb_init = EmbeddingInitializer(self.configs["emb_path"], vocabulary=vocabulary)
        emb_mat = emb_init.get_embedding()
        self.word_emb_pretrained.weight.data.copy_(torch.Tensor(emb_mat))
        self.word_emb_pretrained.weight.requires_grad = self.configs["train"]

    def get_embedding_pretrained(self):
        return self.word_emb_pretrained

    @staticmethod
    def default_configs():
        return {
            "emb_path": None,
            "train": False,
        }


class EmbeddingInitializer:
    def __init__(self, emb_path, vocabulary: Vocabulary):
        self.emb_path = emb_path
        self.vocabulary = vocabulary
        self.vocabulary_list = vocabulary.get_word_list()

        self.word2emb = {}
        self.avg_emb = None
        self.emb_list = []

    def get_embedding(self):
        self._load_embeddings()
        self._constructing_embedding_matrix()
        return np.array(self.emb_list)

    def _load_embeddings(self):
        logger.info("Loading pretrained embeddings from {}.".format(self.emb_path))
        with open(self.emb_path, encoding="utf-8") as f:
            if self.emb_path.endswith("word"):
                f.readline()
            for line in f:
                line = line.strip().split(" ")
                word = line[0]
                if word in self.vocabulary_list:
                    self.word2emb[word] = np.array([float(l) for l in line[1:]])

    def _constructing_embedding_matrix(self):
        """
        Initialize the embedding for out-of-vocabulary words by average embedding.
        """
        logger.info("Constructing word embedding matrix.")
        self._get_avg_emb()

        cnt_oov = 0
        cnt_iv = 0
        self.word2emb[self.vocabulary.pad_token] = np.zeros(300, dtype="float")
        self.emb_list.append(self.word2emb[self.vocabulary.pad_token])
        for i in range(1, len(self.vocabulary_list)):   # skip [PAD]
            word = self.vocabulary.get_word(i)
            if word not in self.word2emb:
                cnt_oov += 1
                self.word2emb[word] = self.avg_emb
                self.emb_list.append(self.avg_emb)
            else:
                cnt_iv += 1
                self.emb_list.append(self.word2emb[word])
        logger.info("Words in vocabulary: {},  out of vocabulary: {}".format(cnt_iv, cnt_oov))

    def _get_avg_emb(self):
        emb_list = list(self.word2emb.values())
        self.avg_emb = np.mean(emb_list, axis=0)
