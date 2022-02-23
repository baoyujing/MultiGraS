import os
import re
import time
import torch
import shutil
import datetime
from multiprocessing import Pool

from pyrouge import Rouge155

import sys
sys.setrecursionlimit(10000)

_ROUGE_PATH = "/home/bruce/Desktop/text_graph_sum/evaluators/ROUGE-1.5.5"
_PYROUGE_TEMP_FILE = "/home/bruce/Desktop/text_graph_sum/tmp"

REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}", "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}


def clean(x):
    x = x.lower()
    return re.sub(
            r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
            lambda m: REMAP.get(m.group()), x)


def pyrouge_score_all(hyps_list, refer_list):
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    PYROUGE_ROOT = os.path.join(_PYROUGE_TEMP_FILE, nowTime)
    SYSTEM_PATH = os.path.join(PYROUGE_ROOT, 'result')
    MODEL_PATH = os.path.join(PYROUGE_ROOT, 'gold')
    if os.path.exists(SYSTEM_PATH):
        shutil.rmtree(SYSTEM_PATH)
    os.makedirs(SYSTEM_PATH)
    if os.path.exists(MODEL_PATH):
        shutil.rmtree(MODEL_PATH)
    os.makedirs(MODEL_PATH)

    assert len(hyps_list) == len(refer_list)

    for i in range(len(hyps_list)):
        system_file = os.path.join(SYSTEM_PATH, 'Model.%d.txt' % i)
        model_file = os.path.join(MODEL_PATH, 'Reference.A.%d.txt' % i)

        # refer = clean(refer_list[i])
        # hyps = clean(hyps_list[i])
        refer = refer_list[i]
        hyps = hyps_list[i]

        with open(system_file, 'wb') as f:
            f.write(hyps.encode('utf-8'))
        with open(model_file, 'wb') as f:
            f.write(refer.encode('utf-8'))

    r = Rouge155(_ROUGE_PATH)

    r.system_dir = SYSTEM_PATH
    r.model_dir = MODEL_PATH
    r.system_filename_pattern = 'Model.(\d+).txt'
    r.model_filename_pattern = 'Reference.[A-Z].#ID#.txt'

    try:
        # output = r.convert_and_evaluate(rouge_args="-e %s -a -m -n 4" % os.path.join(_ROUGE_PATH, "data"))
        # output = r.convert_and_evaluate(rouge_args="-e %s -a" % os.path.join(_ROUGE_PATH, "data"))
        output = r.convert_and_evaluate()
        output_dict = r.output_to_dict(output)
    except:
        print("[ERROR] Error stop, delete PYROUGE_ROOT...")
        shutil.rmtree(PYROUGE_ROOT)
    print(output_dict)
    scores = {}
    scores['rouge-1'], scores['rouge-2'], scores['rouge-l'] = {}, {}, {}
    scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f'] = output_dict['rouge_1_precision'], \
                                                                             output_dict['rouge_1_recall'], output_dict[
                                                                                 'rouge_1_f_score']
    scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f'] = output_dict['rouge_2_precision'], \
                                                                             output_dict['rouge_2_recall'], output_dict[
                                                                                 'rouge_2_f_score']
    scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f'] = output_dict['rouge_l_precision'], \
                                                                             output_dict['rouge_l_recall'], output_dict[
                                                                                 'rouge_l_f_score']
    return scores


def process(data):
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = "rouge-tmp-{}-{}".format(current_time,pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w", encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w", encoding="utf-8") as f:
                f.write(references[i])
        r = Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def test_rouge(cand, ref, num_processes):
    """Calculate ROUGE scores of sequences passed as an iterator
       e.g. a list of str, an open file, StringIO or even sys.stdin
    """
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]

    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)
    candidates_chunks = list(chunks(candidates, int(len(candidates)/num_processes)))
    references_chunks = list(chunks(references, int(len(references)/num_processes)))
    n_pool = len(candidates_chunks)
    arg_lst = []
    for i in range(n_pool):
        arg_lst.append((candidates_chunks[i], references_chunks[i], i))
    pool = Pool(n_pool)
    results = pool.map(process, arg_lst)
    final_results = {}
    for i, r in enumerate(results):
        for k in r:
            if k not in final_results:
                final_results[k] = r[k]*len(candidates_chunks[i])
            else:
                final_results[k] += r[k] * len(candidates_chunks[i])
    for k in final_results:
        final_results[k] = final_results[k]/len(candidates)
    return final_results


def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        )


def output2doc_extractive(outputs, len_list, original_document, original_summary, blocking=False, n_win=3, index_list=None):
    hyp_list = []
    ref_list = []
    start_pos = 0
    for i, n_sent in enumerate(len_list):
        prediction = outputs[start_pos:start_pos+n_sent].max(1)[1]    # [node]
        if index_list[i] == 0:  # CNN
            k = 2
        else:
            k = 3   # DM
        if blocking:
            pred_idx = ngram_blocking(sents=original_document[i],
                                      p_sent=outputs[start_pos:start_pos+n_sent, 1], n_win=n_win, k=k)
        else:
            if k == 0:
                pred_idx = torch.arange(n_sent)[prediction != 0].long()
            else:  # select top
                topk, pred_idx = torch.topk(outputs[start_pos:start_pos+n_sent, 1], min(k, n_sent))
        hyp = label2str(labels=pred_idx, document=original_document[i])
        ref = doc2str(document=original_summary[i])
        hyp_list.append(hyp)
        ref_list.append(ref)
        start_pos += n_sent
    return hyp_list, ref_list


def ngram_blocking(sents, p_sent, n_win=3, k=3):
    """
    p_sent: [sent_num, 1]
    n_win: int, n_win=2,3,4...
    """
    ngram_list = []
    _, sorted_idx = p_sent.sort(descending=True)
    sents_filtered = []
    for idx in sorted_idx:
        pieces = sents[idx]
        overlap_flag = 0
        sent_ngram = []
        for i in range(len(pieces) - n_win):
            ngram = " ".join(pieces[i: (i + n_win)])
            if ngram in ngram_list:
                overlap_flag = 1
                break
            else:
                sent_ngram.append(ngram)
        if overlap_flag == 0:
            sents_filtered.append(idx)
            ngram_list.extend(sent_ngram)
            if len(sents_filtered) >= k:
                break
    sents_filtered = torch.LongTensor(sents_filtered)
    return sents_filtered


def doc2str(document):
    """
    document: list of list [n_sent, n_word]
    """
    document = [" ".join(d) for d in document]
    return "\n".join(document)


def label2str(labels, document):
    doc = [document[l] for l in labels]
    return doc2str(doc)


def label2str_pad(labels, document):
    doc = [document[l] for i, l in enumerate(labels) if not (i > 0 and l != 0)]
    return doc2str(doc)


def get_oracle_sents(label_batch, document_batch):
    sent_list = []
    for i, labels in enumerate(label_batch):
        sent_list.append(label2str_pad(labels=label_batch[i], document=document_batch[i]))
    return sent_list
