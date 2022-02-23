import os
import time
import json
import logging

from preprocessors.utils import hashhex

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def split_data(split_root, save_root):
    logger.info("Splitting data.")
    start_time = time.time()

    if not os.path.exists(split_root):
        os.makedirs(split_root)

    data_list = _read(path=os.path.join(split_root, "cnn_wayback_training_urls.txt"))
    _write(data_list, save_path=os.path.join(save_root, "cnn_train.txt"))

    data_list = _read(path=os.path.join(split_root, "cnn_wayback_validation_urls.txt"))
    _write(data_list, save_path=os.path.join(save_root, "cnn_val.txt"))

    data_list = _read(path=os.path.join(split_root, "cnn_wayback_test_urls.txt"))
    _write(data_list, save_path=os.path.join(save_root, "cnn_test.txt"))

    logger.info("Time elapsed: {:.2f}s.".format(time.time() - start_time))


def _read(path):
    split_list = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip().encode("utf-8")
            hashid = hashhex(s)
            split_list.append(hashid)
    return split_list


def _write(data_list, save_path):
    with open(save_path, "w") as f:
        for line in data_list:
            f.writelines(line + ".json\n")


if __name__ == "__main__":
    split_root = "../data/cnn_dailymail/split/"
    save_path = "../data/dataset_processed_test/cnn_dailymail/split"

    split_data(split_root, save_path)
