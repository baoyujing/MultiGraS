import time
import json
import torch
import logging
import argparse
from tqdm import tqdm

from modules.multi_gras import MultiGraS
from utils import test_rouge, rouge_results_to_str, output2doc_extractive
from data_management.vocabulary import Vocabulary
from data_management.data_loader import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, model, data_loader, cnn_eval_path, cuda=True):
        if cuda:
            self.model = model.cuda()

        self.data_loader = data_loader
        self.cuda = cuda

        self.cnn_eval_path = cnn_eval_path
        self.cnn_files = []
        with open(self.cnn_eval_path) as f:
            for line in f:
                self.cnn_files.append(line.strip().split(".")[0])

    def eval(self):
        logger.info("Start evaluation.")
        start_time = time.time()
        hyp_list = []
        ref_list = []

        with torch.no_grad():
            self.model.eval()
            for i, pack in enumerate(tqdm(self.data_loader)):
                if self.cuda:
                    pack["document"] = pack["document"].cuda()
                    pack["graphs"] = pack["graphs"].cuda()
                    pack["graphs_sent"] = [adj.cuda() for adj in pack["graphs_sent"]]

                    outputs = self.model(doc_input=pack["document"],
                                         sent_len_list=pack["sent_len_list"],
                                         adjs=pack["graphs"],
                                         n_sent_list=pack["n_sent"],
                                         adjs_sents=pack["graphs_sent"])
                hashid_list = pack["hashid_list"]
                index_list = self.get_index(hashid_list)

                hyps, refs = output2doc_extractive(
                    outputs=outputs,
                    len_list=pack["n_sent"],
                    original_document=pack["original_document"],
                    original_summary=pack["original_summary"],
                    blocking=True, n_win=3, index_list=index_list)
                hyp_list.extend(hyps)
                ref_list.extend(refs)

        scores_all = test_rouge(cand=hyp_list, ref=ref_list, num_processes=16)
        scores_all = rouge_results_to_str(scores_all)

        logger.info('Test completed! Time elapsed: {:.2f}s.'.format(time.time() - start_time))
        return scores_all

    def get_index(self, hashid_list):
        index_list = []
        for hashid in hashid_list:
            if hashid in self.cnn_files:
                index_list.append(0)
            else:
                index_list.append(1)
        return index_list


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./ckpts/0.ckpt")
parser.add_argument("--save_path", type=str, default="./scores_test.txt")
parser.add_argument("--cnn_eval_path", type=str, default="./data/dataset_processed/cnn_dailymail/split/cnn_test.txt")
parser.add_argument("--config_model", type=str, default="./configs/model.yml")
parser.add_argument("--config_dataloader", type=str, default="./configs/dataloader.yml")
parser.add_argument("--config_vocabulary", type=str, default="./configs/vocabulary.yml")

args = parser.parse_args()
vocabulary = Vocabulary(configs_path=args.config_vocabulary)
model = MultiGraS(configs_path=args.config_model, vocabulary=vocabulary)
pretrained_dict = torch.load(args.model_path)
model.load_state_dict(pretrained_dict)

loader = DataLoader(config_path=args.config_dataloader, vocabulary=vocabulary, split="test")
evaluator = Evaluator(model=model, data_loader=loader, cnn_eval_path=args.cnn_eval_path)
scores = evaluator.eval()

f = open(args.save_path, "w")
f.writelines(json.dumps(scores))
f.writelines("\n\n")
f.close()
