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
        pairs = [pair for pair in pairs if len(pair[0]) >= min_length and len(pair[0]) <= max_length \
                 and len(pair[1]) >= min_length and len(pair[1]) <= max_length]

    # print("Applied filter by length, min sentence length %s, max sentence length %s" % (min_length, max_length))

    if filter_func:
        pairs = [pair for pair in pairs if filter_func(pair[0]) or filter_func(pair[1])]

    return pairs


def train_split(pairs, test_ratio=0.2):
    num_data = len(pairs)
    indices = list(range(num_data))

    # Randomly splitting indices:
    val_len = int(np.floor(test_ratio*num_data))
    val_idx = np.random.choice(indices, size=val_len, replace=False)

    test_len = int(val_len*0.50)
    test_idx = np.random.choice(val_idx, size=test_len, replace=False)

    val_idx = list(set(val_idx) - set(test_idx))

    train_idx = list(set(indices) - set(val_idx))

    print( len(val_idx) + len(train_idx) + len(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

   # train_size = int(train_ratio * num_data)
   # val_size = num_data - train_size # 20
   # train_dataset, val_dataset = torch.utils.data.random_split(pairs, [train_size, val_size])
    return train_sampler, validation_sampler, test_sampler


### train / testing utils

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = - torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()