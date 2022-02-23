import re
import time
import json
import logging
from multiprocessing import Pool

from preprocessors.utils import chunk_list, get_paths_of_root


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OracleBuilder:
    def __init__(self, summary_size=3, n_process=12):
        self.summary_size = summary_size
        self.n_process = n_process   # the number of processors

    """
    Multi-processor
    """

    def extract_oracle_mp(self, data_root):
        """
        Use multiple processors to extract oracles.
        :param data_root: root to .json files
        """
        logger.info("Extracting oracles for data in {}.".format(data_root))
        start_time = time.time()

        file_paths = get_paths_of_root(data_root=data_root)
        file_chunks = chunk_list(file_paths, n_process=self.n_process)
        inputs = [(i, file_chunk) for i, file_chunk in enumerate(file_chunks)]

        pool = Pool(self.n_process)
        pool.map(self._extract_oracle_summary, inputs)
        pool.close()
        pool.join()

        logger.info("Complete! Total time elapsed: {:.2f}s.".format(time.time() - start_time))

    def _extract_oracle_summary(self, inputs):
        worker_id, file_chunk = inputs
        start_time = time.time()
        for i, fp in enumerate(file_chunk):
            if (i + 1) % 10000 == 0:
                logger.info("Worker {}: {} documents processed, {:.2f}s elapsed.".format(worker_id, i+1, time.time()-start_time))
            data = json.load(open(fp))
            sum_sent_ids = self.greedy_selection(data["document"], data["summary"], self.summary_size)
            data["oracle_summary"] = sum_sent_ids
            json.dump(data, open(fp, "w"))
        logger.info("Worker {}: {} documents processed in total, {:.2f}s elapsed total.".format(
            worker_id, len(file_chunk), time.time() - start_time))

    def greedy_selection(self, doc_sent_list, abstract_sent_list, summary_size=3):
        max_rouge = 0.0
        abstract = sum(abstract_sent_list, [])
        abstract = self._rouge_clean(' '.join(abstract)).split()
        sents = [self._rouge_clean(' '.join(s)).split() for s in doc_sent_list]
        evaluated_1grams = [self._get_word_ngrams(1, [sent]) for sent in sents]
        reference_1grams = self._get_word_ngrams(1, [abstract])
        evaluated_2grams = [self._get_word_ngrams(2, [sent]) for sent in sents]
        reference_2grams = self._get_word_ngrams(2, [abstract])

        selected = []
        for s in range(summary_size):
            cur_max_rouge = max_rouge
            cur_id = -1
            for i in range(len(sents)):
                if (i in selected):
                    continue
                c = selected + [i]
                candidates_1 = [evaluated_1grams[idx] for idx in c]
                candidates_1 = set.union(*map(set, candidates_1))
                candidates_2 = [evaluated_2grams[idx] for idx in c]
                candidates_2 = set.union(*map(set, candidates_2))
                rouge_1 = self._cal_rouge(candidates_1, reference_1grams)['f']
                rouge_2 = self._cal_rouge(candidates_2, reference_2grams)['f']
                rouge_score = rouge_1 + rouge_2
                if rouge_score > cur_max_rouge:
                    cur_max_rouge = rouge_score
                    cur_id = i
            if (cur_id == -1):
                return selected
            selected.append(cur_id)
            max_rouge = cur_max_rouge
        return selected

    def _get_ngrams(self, n, text):
        """
        Calcualtes n-grams.
        :param n: which n-grams to calculate
        :param text: An array of tokens
        :return: A set of n-grams
        """
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _get_word_ngrams(self, n, sentences):
        """
        Calculates word n-grams for multiple sentences.
        """
        assert len(sentences) > 0
        assert n > 0
        words = sum(sentences, [])
        return self._get_ngrams(n, words)

    def _rouge_clean(self, s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    def _cal_rouge(self, evaluated_ngrams, reference_ngrams):
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
        return {"f": f1_score, "p": precision, "r": recall}
