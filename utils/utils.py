import random

import numpy
import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler

from global_settings import device

# Practical PyTorch and PyTorch Tutorials
# If preprocessing is done with "expand_contractions" flag set to True,
# the prefixes right part, e.g. 'we re', should not appear in the corpus
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def max_length(lines):
    return max(len(line.split()) for line in lines)


def english_prefixes(sentence):
    return sentence.startswith(eng_prefixes)


""" Function at pair level """


def filter_pairs(pairs, len_tuple=None, filter_func=None):
    """
    :param pairs: The sentence pair
    :param len_tuple: A tuple including a min_len and a max_len, e.g. (3,25)
    :param filter_func: A tuple, including boolean filter functions, e.g. filter_sentence_by_prefix
    :return: the filtered pairs and the size of the filtered dataset
    """
    if len_tuple:
        min_length = len_tuple[0]
        max_length = len_tuple[1]
        pairs = [pair for pair in pairs if len(pair[0]) >= min_length and len(pair[0]) <= max_length]

    # print("Applied filter by length, min sentence length %s, max sentence length %s" % (min_length, max_length))

    if filter_func:
        pairs = [pair for pair in pairs if filter_func(pair[0]) or filter_func(pair[1])]

    return pairs


### train / testing utils

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def split_data(data, test_ratio=0.1):
    num_samples = len(data)
    test_range = int(num_samples*test_ratio)
    train_range = num_samples-test_range
    random.shuffle(data)

    train_set = data[:train_range]
    test_set = data[train_range:]
    return train_set, test_set
