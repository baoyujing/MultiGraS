import yaml
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, configs_path):
        configs = yaml.safe_load(open(configs_path))
        self.configs = self.default_configs()
        self.configs.update(configs)

        self.vocabulary_path = self.configs["vocabulary_path"]
        self.max_size = self.configs["max_size"]  # The maximum size of the resulting Vocabulary.
        self.unk_token = self.configs["unk_token"]
        self.pad_token = self.configs["pad_token"]

        self._word2id = {
            self.configs["pad_token"]: self.configs["pad_id"],
            self.configs["unk_token"]: self.configs["unk_id"],
        }
        self._id2word = {
            self.configs["pad_id"]: self.configs["pad_token"],
            self.configs["unk_id"]: self.configs["unk_token"],
        }
        self._n_words = 2
        self._load_vocabulary()

    def _load_vocabulary(self):
        logger.info("Build vocabulary.")

        with open(self.vocabulary_path) as f:
            cnt = 0
            for line in f:
                cnt += 1
                w = line.strip().split("\t")[0]
                if w in self._word2id:
                    continue
                self._word2id[w] = self._n_words
                self._id2word[self._n_words] = w
                self._n_words += 1
                if self._n_words >= self.max_size:
                    logger.info("{} words loaded. Stop reading.".format(self._n_words))
                    break

    def get_id(self, word):
        if word not in self._word2id:
            return self._word2id[self.unk_token]
        return self._word2id[word]

    def get_word(self, word_id):
        if word_id not in self._id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id2word[word_id]

    def __len__(self):
        return self._n_words

    def get_word_list(self):
        return self._word2id.keys()

    @staticmethod
    def default_configs():
        return {
            "vocabulary_path": "../dataset_processed/cnn_dailymail/vocabulary.txt",
            "max_size": 50000,
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "pad_id": 0,
            "unk_id": 1,
        }


if __name__ == "__main__":
    path = "../configs/vocabulary.yml"
    vocabulary = Vocabulary(configs_path=path)
    print(vocabulary.get_id("[UNK]"))
    print(vocabulary.get_id("cnn"))
    print(len(vocabulary))
    print(vocabulary.get_word_list())
