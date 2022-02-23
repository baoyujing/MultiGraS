import os
import yaml
import time
import json
import torch
import logging
import argparse
from tqdm import tqdm

from modules.multi_gras import MultiGraS
from data_management.vocabulary import Vocabulary
from data_management.data_loader import DataLoader
from evaluate import Evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, configs_path, model, loader_train, loader_val):
        self.configs = self.default_configs()
        configs = yaml.safe_load(open(configs_path))
        self.configs.update(configs)

        self.model = model
        if self.configs["cuda"]:
            self.model = model.cuda()
        if self.configs["pretrained_path"] is not None:
            self.model.load_state_dict(torch.load(self.configs["pretrained_path"]))
        self.loader_train = loader_train
        self.loader_val = loader_val

        self.optimizer = torch.optim.Adam(model.parameters(), lr=float(self.configs["lr"]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")

        self.save_root = self.configs["save_root"]
        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root)

    def train(self):
        start_time = time.time()
        cnt_batch = 0
        f = open("results_eval.txt", "w")
        f2 = open("results_loss.txt", "w")
        for epoch in range(self.configs["n_epoch"]):
            logger.info("Epoch: {}".format(epoch))
            self.model.train()
            epoch_loss = 0.0

            for i, pack in enumerate(tqdm(self.loader_train)):
                start_time = time.time()
                self.optimizer.zero_grad()

                if self.configs["cuda"]:
                    pack["document"] = pack["document"].cuda()
                    pack["graphs"] = pack["graphs"].cuda()
                    pack["graphs_sent"] = [adj.cuda() for adj in pack["graphs_sent"]]

                outputs = self.model(doc_input=pack["document"],
                                     sent_len_list=pack["sent_len_list"],
                                     adjs=pack["graphs"],
                                     n_sent_list=pack["n_sent"],
                                     adjs_sents=pack["graphs_sent"])

                labels = self.get_labels(pack["oracle_summary"], pack["n_sent"])
                loss = self.criterion(outputs, labels)
                loss = loss.mean()

                loss.backward()

                self.optimizer.step()

                batch_loss = loss.detach().cpu().numpy()
                epoch_loss += batch_loss
                torch.cuda.empty_cache()

                cnt_batch += 1

            logger.info("Epoch: {}, loss: {:.2f}, time elapsed: {:.2f}s.".
                        format(epoch, epoch_loss/len(self.loader_train), time.time() - start_time))
            f2.writelines(str(epoch_loss))
            f2.writelines("\n")
            f2.flush()

            # save model
            path = os.path.join(self.save_root, "{}.ckpt".format(epoch))
            torch.save(self.model.state_dict(), path)

            # evaluate
            scores = self.eval()
            f.writelines(json.dumps(scores))
            f.writelines("\n")
            f.flush()

    def get_labels(self, oracle_summary, len_list):
        """
        oracle_summary: [oracles]
        :return labels: [n] 0 or 1
        """
        labels = []
        for i, idx in enumerate(oracle_summary):
            label = torch.zeros(len_list[i], dtype=torch.long)
            indices = oracle_summary[i]
            label[indices] = 1
            labels.append(label)
        labels = torch.cat(labels)
        if self.configs["cuda"]:
            return labels.to("cuda")
        return labels

    def eval(self):
        evaluator = Evaluator(model=self.model, data_loader=self.loader_val, cuda=self.configs["cuda"],
                              cnn_eval_path=self.configs["cnn_eval_path"])
        scores_all = evaluator.eval()
        return scores_all

    @staticmethod
    def default_configs():
        return {
            "lr": 5e-4,
            "batch_size": 32,
            "n_epoch": 10,
            "save_root": "./ckpts",
            "pretrained_path": None,
            "cuda": True,
            "cnn_split_path": None
        }


parser = argparse.ArgumentParser()
parser.add_argument("--config_trainer", type=str, default="./configs/trainer.yml")
parser.add_argument("--config_model", type=str, default="./configs/model.yml")
parser.add_argument("--config_dataloader", type=str, default="./configs/dataloader.yml")
parser.add_argument("--config_vocabulary", type=str, default="./configs/vocabulary.yml")

args = parser.parse_args()
vocabulary = Vocabulary(configs_path=args.config_vocabulary)
model = MultiGraS(configs_path=args.config_model, vocabulary=vocabulary)
loader_train = DataLoader(config_path=args.config_dataloader, vocabulary=vocabulary, split="train")
loader_val = DataLoader(config_path=args.config_dataloader, vocabulary=vocabulary, split="val")

trainer = Trainer(configs_path=args.config_trainer, model=model, loader_train=loader_train, loader_val=loader_val)
trainer.train()
