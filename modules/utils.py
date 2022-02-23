import torch


def get_feat_doc(doc_inputs, len_list):
    feature_list = []
    pos = 0
    for n in len_list:
        feature_list.append(doc_inputs[pos:pos+n])
        pos += n
    return feature_list


def glen2mask(glen_list, max_len):
    mask_list = []
    for glen in glen_list:
        mask = torch.arange(max_len) >= glen
        mask_list.append(mask)
    return torch.stack(mask_list)
