import os
import time
import json
import logging
from multiprocessing import Pool
from preprocessors.utils import chunk_list, get_paths_of_root, get_paths_of_split
from preprocessors.nlp_pipeline import Pipeline

from preprocessors.base_processor import BaseProcessor
from preprocessors.cnndm_reader import Reader
from preprocessors.graph_builder import GraphBuilder

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
# acceptable ways to end a sentence
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]


class Processor(BaseProcessor):
    def __init__(self, config_path):
        super(Processor, self).__init__(config_path=config_path)
        self.cnn_story_root = os.path.join(self.root, self.configs["cnn_story_path"])
        self.dm_story_root = os.path.join(self.root, self.configs["dm_story_path"])

        self.split_root = self.configs["split_root"]
        self.train_path = os.path.join(self.split_root, self.configs["all_train_path"])
        self.val_path = os.path.join(self.split_root, self.configs["all_val_path"])
        self.test_path = os.path.join(self.split_root, self.configs["all_test_path"])

        self.max_n_sents = self.configs["max_n_sents"]
        self.max_n_words = self.configs["max_n_words"]

        self.reader = Reader()

    def process(self):
        logger.info("Start pre-processing")
        self._tokenize_data("cnn")
        self._tokenize_data("dm")
        self._split_data()
        self._build_vocabulary()
        self._extract_oracle()
        self._build_graphs()

    """
    Tokenize
    """

    def _tokenize_data(self, news="cnn"):
        logger.info("Tokenizing {} documents.".format(news))
        if news == "cnn":
            story_root = self.cnn_story_root
            story_file_list = os.listdir(self.cnn_story_root)
        if news == "dm":
            story_root = self.dm_story_root
            story_file_list = os.listdir(self.dm_story_root)

        logger.info("n_stories: {}".format(len(story_file_list)))
        self._mp_helper(story_root=story_root, file_list=story_file_list)

    def _mp_helper(self, story_root, file_list):
        file_list_chunks = chunk_list(list(file_list), self.n_process)
        pool = Pool(self.n_process)

        input_list = []
        for i, file_chunk in enumerate(file_list_chunks):
            input_list.append((i, story_root, file_chunk))
        pool.map(self._process_data, input_list)

        pool.close()
        pool.join()

    def _process_data(self, inputs):
        worker_id, story_root, story_file_list = inputs
        start_time = time.time()
        reader = Reader()
        pipeline = Pipeline(port=worker_id)
        for i, story_file in enumerate(story_file_list):
            if i % 1000 == 0:
                logger.info("Worker: {}, {} documents processed, {:.2f}s elapsed.".format(worker_id, i, time.time()-start_time))
            story_path = os.path.join(story_root, story_file)
            story = reader.read_story(story_path)
            story_tokenized = self._tokenize_story(story=story, pipeline=pipeline)
            if not story_tokenized:  # skip empty files
                continue
            body_sents, body_heads, summary_sents, summary_heads = story_tokenized

            data = {
                "document": body_sents,
                "document_heads": body_heads,
                "summary": summary_sents,
                "summary_heads": summary_heads,
                "hashid": story_file.split(".")[0],
            }

            path = os.path.join(self.save_file_root, data["hashid"] + ".json")
            with open(path, "w") as f:
                json.dump(data, f)
        pipeline.core_nlp.stop()

    def _tokenize_story(self, story, pipeline):
        ret = pipeline(story.strip().lower())

        if not ret:
            return False
        story_tokenized, heads = ret
        if story_tokenized[0][0] == "@highlight":
            return False

        body_sents = []
        body_heads = []
        summary_sents = []
        summary_heads = []
        summary_flag = False
        cnt_high = 0
        for i, sent in enumerate(story_tokenized):
            if sent[0] == "@highlight":
                summary_flag = True
                cnt_high += 1
            elif summary_flag:
                summary_sents.append(self._fix_missing_period(sent))
                summary_heads.append(heads[i])
            else:
                body_sents.append(self._fix_missing_period(sent))
                body_heads.append(heads[i])
        return body_sents, body_heads, summary_sents, summary_heads

    def _fix_missing_period(self, sent):
        """Adds a period to a line that is missing a period"""
        if sent[-1] in END_TOKENS:
            return sent
        return sent + ["."]

    """
    Split data
    """

    def _split_data(self):
        logger.info("Splitting data.")
        start_time = time.time()

        self._process_split_data(data_path=self.train_path, save_path=self.save_split_train)
        self._process_split_data(data_path=self.val_path, save_path=self.save_split_val)
        self._process_split_data(data_path=self.test_path, save_path=self.save_split_test)

        logger.info("Time elapsed: {:.2f}s.".format(time.time() - start_time))

    def _process_split_data(self, data_path, save_path):
        data_list = self.reader.get_split(path=data_path)

        with open(save_path, "w") as f:
            for line in data_list:
                f.writelines(line + ".json\n")

    """
    Build Tf-idf graph
    """

    def _build_graphs(self):
        self._update_stop_words()
        self.graph_builder = GraphBuilder(
            stop_word_list=self.stop_words_list,
            min_df=self.configs["min_df"],
            n_process=self.n_process)
        self._create_graph()

    def _update_stop_words(self):
        for word in END_TOKENS:
            self.stop_words_list.append(word)
        self.stop_words_list = list(set(self.stop_words_list))   # remove redundant words


if __name__ == "__main__":
    processor = Processor(config_path="../configs/cnndm_configs.yml")
    processor.process()
